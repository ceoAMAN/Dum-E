from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import configs


# ── Apex-Nadir Convolution ──────────────────────────────────────────────────
# Nadir is the UNDERFIT boundary, apex the OVERFIT boundary, and the goldilocks
# allocation sits between them. Nadir is NOT a discard gate: when tokens are
# scarce we use FEWER experts, never zero. Everything below is derived from T
# (the total tokens received) — no tuned constants.

# The one span requirement ALLOC(T) needs to be identifiable. Fitting a slope in
# log-log across T values that differ by less than this ratio is fitting noise:
# measured on a x2.2 span the exponent came out +1.99 on one batch and -1.02 on
# the next. e is the smallest ratio that gives the regressor a full unit of
# log-range to work with, so it is the natural floor rather than a tuned one.
SPAN_MIN_RATIO = math.e
@dataclass
class ProbeSizes:
    """The three allocations probed per input, all functions of T alone:
        n     : natural number with n^2 < T < (n+1)^2      (n = isqrt)
        lo    = T mod n^2        — deep in the underfit region, anchors nadir
        mid   = sqrt(T^2 - lo^2) / 2
        hi    = T                — the whole input to one expert, anchors apex
    """
    lo: int
    mid: int
    hi: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.lo, self.mid, self.hi)


def probe_sizes(total_tokens: int) -> ProbeSizes:
    t = int(total_tokens)
    if t <= 1:
        return ProbeSizes(lo=1, mid=1, hi=max(1, t))
    n = math.isqrt(t)
    if n * n >= t:          # need n^2 STRICTLY below T
        n -= 1
    n = max(1, n)
    lo = t % (n * n) if n > 1 else t - 1
    lo = max(1, min(lo, t))          # a 0-token probe cannot be executed at all
    mid = int(round(math.sqrt(max(0.0, float(t) * t - float(lo) * lo)) / 2.0))
    # sqrt(T^2 - lo^2)/2 drops BELOW lo once lo > T/sqrt(5), which happens only for
    # T < ~20 (lo is bounded by 2*sqrt(T)). Clamp so lo <= mid <= hi always holds;
    # for such tiny inputs the three probes simply collapse toward each other.
    mid = max(lo, min(mid, t))
    return ProbeSizes(lo=lo, mid=mid, hi=t)


@dataclass
class ProbeRecord:
    """One measurement: expert `expert_id` was fed `probe_size` tokens taken from
    an input of `total_tokens`, and scored `quality`. Degenerate measurements
    (empty generation / ~zero-norm hidden state / NaN) must be DROPPED by the
    caller rather than recorded as quality 0 — a fabricated zero lands exactly in
    the low-t region the nadir is fitted from and drags the curve down where we
    have the least real information."""
    expert_id: int
    total_tokens: int
    probe_size: int
    quality: float
    wall_time: float = 0.0   # measured cost of this probe; 0.0 = not recorded


@dataclass
class AllocationModel:
    """ALLOC(T): tokens-per-expert as a power law in T, fitted over the goldilocks
    points produced by the paired convolution. Power law (fit in log-log) because
    it is scale-free and stays sublinear — a polynomial extrapolates absurdly the
    moment a bigger input than anything calibrated arrives."""
    log_a: float = 0.0
    b: float = 1.0
    fitted: bool = False
    t_max_seen: int = 0
    n_points: int = 0

    def predict(self, total_tokens: int) -> float:
        t = max(1, int(total_tokens))
        if not self.fitted:
            return float(t)          # uncalibrated: one expert takes the input
        return float(min(t, max(1.0, math.exp(self.log_a) * (t ** self.b))))


@dataclass
class ExpertCurves:
    apex_coeffs: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, -0.001]))
    nadir_coeffs: np.ndarray = field(default_factory=lambda: np.array([float(configs.FRAGMENT_MIN), 0.0]))
    latency_coeffs: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.001]))
    apex_ceiling: float = 256.0
    nadir_floor: float = float(configs.FRAGMENT_MIN)
    r_out_cached: Optional[float] = None
    # True once this expert's curves have been fitted from real data (calibration
    # or live latency). While False the expert is "cold" — compute_r_out reports
    # no estimate so the caller falls back to EXPERT_BOOTSTRAP_TOKENS for the first
    # run, then apex-nadir takes over.
    has_data: bool = False
    # Sufficient statistics for a two-parameter least-squares latency fit:
    # (n, sum_t, sum_t2, sum_w, sum_tw). Five floats instead of a ring buffer —
    # the normal equations are exact and incremental, so update_latency stays
    # O(1) and nothing has to remember individual probes.
    lat_stats: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
class ApexNadirConvolution:
    def __init__(self, calibration_path: str, latency_store_path: str):
        self.calibration_path = calibration_path
        self.latency_store_path = latency_store_path
        self.expert_curves: Dict[int, ExpertCurves] = {}
        # Apex-nadir calibration state: raw probes, the goldilocks point derived
        # per input, and the fitted ALLOC(T) that generalises across input sizes.
        self.probe_records: List[ProbeRecord] = []
        self.goldilocks_points: List[Tuple[int, float, float]] = []   # (T, alloc, spread)
        self.alloc = AllocationModel()
        # Central's own apex-nadir result: the smallest input it handles well
        # ALONE. training.probe_central_capacity() measures it; the A/B decision
        # keys on it. 0.0 = not yet probed.
        self.central_min_cap: float = 0.0
        self._init_default_curves()

    # ── probing ────────────────────────────────────────────────────────────
    def record_probe(self, expert_id: int, total_tokens: int, probe_size: int,
                     quality: Optional[float], wall_time: float = 0.0) -> bool:
        """Store one probe measurement. `quality=None` (or non-finite) means the
        measurement was degenerate and is DISCARDED — see ProbeRecord.
        `wall_time` is what makes the goldilocks efficiency-aware; without it the
        convolution maximises quality alone and always prefers a bigger share."""
        if quality is None or not math.isfinite(float(quality)):
            return False
        self.probe_records.append(ProbeRecord(
            expert_id=int(expert_id), total_tokens=int(total_tokens),
            probe_size=int(probe_size), quality=float(quality),
            wall_time=float(wall_time) if math.isfinite(float(wall_time)) else 0.0,
        ))
        return True

    def _records_for(self, total_tokens: int) -> Dict[int, List[ProbeRecord]]:
        by_expert: Dict[int, List[ProbeRecord]] = {}
        for r in self.probe_records:
            if r.total_tokens == int(total_tokens):
                by_expert.setdefault(r.expert_id, []).append(r)
        return by_expert

    @staticmethod
    def _fit_cost(cost_by_size: Dict[int, List[float]], grid: np.ndarray) -> Optional[np.ndarray]:
        """Measured cost over the grid as `a + b*t`, from the sample's wall times.
        The INTERCEPT is the point: a per-forward fixed cost exists (dispatch,
        weight access, KV setup) and attributing it to the slope is what makes a
        quality/cost ratio monotonically decreasing, collapsing the optimum onto
        the floor. Returns None when no wall times were recorded, in which case
        the caller falls back to pure grounded quality."""
        if len(cost_by_size) < 2:
            return None
        sizes = np.array(sorted(cost_by_size), dtype=np.float64)
        costs = np.array([float(np.median(cost_by_size[int(s)])) for s in sizes])
        if not np.all(np.isfinite(costs)) or float(np.max(costs)) <= 0.0:
            return None
        try:
            b, a = np.polyfit(sizes, costs, 1)
        except Exception:
            return None
        floor = float(np.min(costs[costs > 0.0])) * 1e-3
        return np.clip(a + b * grid, floor, None)

    @staticmethod
    def _fit_curve(points: Sequence[ProbeRecord]) -> Optional[np.ndarray]:
        """Quadratic through an expert's probe points (exact with the 3 probes).
        Returns coeffs for np.polyval, or None if the points are degenerate."""
        xs = np.array([p.probe_size for p in points], dtype=np.float64)
        ys = np.array([p.quality for p in points], dtype=np.float64)
        if len(np.unique(xs)) < 2:
            return None
        deg = 2 if len(np.unique(xs)) >= 3 else 1
        try:
            return np.polyfit(xs, ys, deg)
        except Exception:
            return None

    # ── ranking, pairing, convolution ──────────────────────────────────────
    def rank_experts(self, total_tokens: int) -> Tuple[List[int], List[int]]:
        """Rank every probed expert by mean quality across its probe sizes, then
        take the sqrt(E) best and sqrt(E) worst. sqrt(E) of a 100-expert pool is
        10 — i.e. the 90th/10th percentile envelopes, so no single lucky batch can
        define a curve (the failure mode of fitting raw min/max)."""
        by_expert = self._records_for(total_tokens)
        if len(by_expert) < 2:
            return ([], [])
        means = sorted(
            ((eid, float(np.mean([p.quality for p in recs]))) for eid, recs in by_expert.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        k = max(1, int(math.isqrt(len(means))))
        best = [eid for eid, _ in means[:k]]
        worst = [eid for eid, _ in means[-k:]][::-1]   # worst-first, to rank-match
        return (best, worst)

    def convolve_input(self, total_tokens: int) -> Optional[Tuple[float, float]]:
        """TWO regressions over the SAME 2*sqrt(E) sample, then convolve them.

            nadir(t)  — the UNDERFITTING curve, fitted from the low-token regime
            apex(t)   — the OVERFITTING curve,  fitted from the high-token regime
            g(t)      = sqrt( apex(t) * nadir(t) )
            goldilocks = argmax g(t)

        Both the sqrt(E) best AND the sqrt(E) worst feed BOTH curves. Best/worst
        selects the SAMPLE and gives the envelope its spread; it does not mean
        "best -> apex, worst -> nadir". Apex and nadir are token-count regimes —
        too many vs too few — and every sampled expert exhibits both, so splitting
        them by expert rank would confuse which-expert with which-failure-mode.

        Both curves are fitted across ALL THREE probe sizes, as ENVELOPES of the
        pooled sample: at each probe size, apex takes the sample's upper quantile
        and nadir its lower quantile. Splitting the regimes by cutting at the mid
        probe instead was measurably worse — it leaves only two distinct sizes per
        side, so each curve degenerates to a straight line, and a rising line
        times a falling line peaks at the grid EDGE rather than in the interior
        (42.9% error vs 1.0% when the true optimum sat at 0.7*T). Three sizes
        admit a quadratic, and a quadratic is what actually locates a peak.

        The geometric mean peaks only where BOTH curves are strong — enough
        tokens to have stopped underfitting, few enough to not yet be overfitting.
        Literal convolution is wrong here: it centres at the SUM of the two peaks,
        which lands past the overfit ceiling the apex exists to enforce.

        `spread` is the std of each sampled expert's OWN 3-point peak — a genuine
        uncertainty measure. It reports PRECISION, not accuracy: the sample can
        agree closely and still be collectively wrong when the true optimum sits
        far from the probes."""
        best, worst = self.rank_experts(total_tokens)
        if not best and not worst:
            return None
        sample = set(best) | set(worst)                # 2*sqrt(E) experts
        by_expert = self._records_for(total_tokens)
        probes = probe_sizes(total_tokens)
        grid = np.linspace(float(probes.lo), float(probes.hi), 128)
        # Quality at each probe size, pooled over the whole sample.
        by_size: Dict[int, List[float]] = {}
        cost_by_size: Dict[int, List[float]] = {}
        peaks: List[float] = []
        for eid in sample:
            recs = by_expert.get(eid, [])
            if not recs:
                continue
            for r in recs:
                by_size.setdefault(r.probe_size, []).append(r.quality)
                if r.wall_time > 0.0:
                    cost_by_size.setdefault(r.probe_size, []).append(r.wall_time)
            c = self._fit_curve(recs)                  # this expert's own peak
            if c is not None:
                v = np.clip(np.polyval(c, grid), 0.0, None)
                if np.any(np.isfinite(v)) and float(np.max(v)) > 0.0:
                    peaks.append(float(grid[int(np.argmax(v))]))
        if len(by_size) < 2:
            return None
        sizes = np.array(sorted(by_size), dtype=np.float64)
        # Quantiles, not min/max: a single lucky or unlucky expert must not
        # define an entire envelope.
        hi_q = np.array([np.quantile(by_size[int(s)], 0.9) for s in sizes])
        lo_q = np.array([np.quantile(by_size[int(s)], 0.1) for s in sizes])
        mid_q = np.array([np.quantile(by_size[int(s)], 0.5) for s in sizes])
        deg = 2 if len(sizes) >= 3 else 1
        try:
            apex_c = np.polyfit(sizes, hi_q, deg)      # OVERFIT envelope  (p90)
            nadir_c = np.polyfit(sizes, lo_q, deg)     # UNDERFIT envelope (p10)
            mid_c = np.polyfit(sizes, mid_q, deg)      # GROUNDING curve   (p50)
        except Exception:
            return None
        nadir_v = np.clip(np.polyval(nadir_c, grid), 0.0, None)
        apex_v = np.clip(np.polyval(apex_c, grid), 0.0, None)
        mid_v = np.clip(np.polyval(mid_c, grid), 0.0, None)
        # Convolve all THREE: the cube root of the product peaks only where the
        # optimistic bound, the pessimistic bound AND the typical case are all
        # strong. The p50 grounding curve is what stops an envelope artefact —
        # one tail of the sample — from setting the allocation on its own.
        g = np.cbrt(apex_v * nadir_v * mid_v)
        # Efficiency: divide by MEASURED cost so the goldilocks is quality per
        # unit of work, not quality at any price. Without this the convolution
        # maximises quality alone and always prefers a larger share.
        #
        # cost is fitted WITH an intercept (a + b*t). Forcing it through the
        # origin is what made the original compute_r_out degenerate: quality
        # rises sublinearly while a through-origin cost rises linearly, so the
        # ratio decreases monotonically and its argmax is always the floor. With
        # a > 0 the ratio has a genuine interior maximum.
        cost_v = self._fit_cost(cost_by_size, grid)
        if cost_v is not None:
            g = g / cost_v
        if not np.any(np.isfinite(g)) or float(np.max(g)) <= 0.0:
            # One side carries no signal — e.g. the true optimum lies below the
            # mid probe, so every overfit-side reading is ~0 and the geometric
            # mean is zero everywhere. Fall back to the SUM, which still peaks
            # wherever the surviving curve does, rather than discarding the input
            # entirely: a biased estimate is worth more than no calibration point,
            # and the caller sees the widened spread.
            g = apex_v + nadir_v + mid_v
            if cost_v is not None:
                g = g / cost_v
            if not np.any(np.isfinite(g)) or float(np.max(g)) <= 0.0:
                return None
        goldilocks = float(grid[int(np.argmax(g))])
        spread = float(np.std(peaks)) if len(peaks) > 1 else 0.0
        return (goldilocks, spread)

    def close_input(self, total_tokens: int) -> Optional[Tuple[float, float]]:
        """Convolve this input's probes into one goldilocks point and keep it as a
        training point for ALLOC(T)."""
        result = self.convolve_input(total_tokens)
        if result is None:
            return None
        alloc, spread = result
        self.goldilocks_points.append((int(total_tokens), alloc, spread))
        self.alloc.t_max_seen = max(self.alloc.t_max_seen, int(total_tokens))
        return result

    # ── ALLOC(T) ───────────────────────────────────────────────────────────
    def fit_allocation(self) -> bool:
        """Regress the goldilocks points against T as a power law (log-log LSQ),
        giving one rule that generalises to input sizes never probed."""
        pts = [(t, a) for t, a, _ in self.goldilocks_points if t > 0 and a > 0]
        if len(pts) < 2:
            return False
        xs = np.log(np.array([t for t, _ in pts], dtype=np.float64))
        ys = np.log(np.array([a for _, a in pts], dtype=np.float64))
        if len(np.unique(xs)) < 2:
            # Every calibration input was the same size: fall back to the mean ratio.
            ratio = float(np.mean(np.exp(ys - xs)))
            self.alloc.log_a = math.log(max(ratio, 1e-9))
            self.alloc.b = 1.0
        else:
            b, log_a = np.polyfit(xs, ys, 1)
            # ADMISSIBILITY — the exponent has a meaning, so it has a range.
            #   b <= 0  allocation SHRINKS as the input grows. A bigger input
            #           giving each expert fewer tokens is not a calibration
            #           result, it is a bad fit.
            #   b >  1  allocation grows faster than the input, so one expert is
            #           handed more tokens than exist.
            # Only 0 < b <= 1 is a statement about the world. Measured on three
            # points spanning T=24..53 the fit returned b=1.992 on one batch and
            # b=-1.024 on the next, because a factor-of-two span cannot identify
            # a slope in log-log. Rejecting the fit and keeping the previous
            # model is the honest response to data that cannot support it —
            # accepting it produced ALLOC(512)=1 and pinned experts_for at K_MAX.
            span = float(np.max(xs) - np.min(xs))
            admissible = (0.0 < float(b) <= 1.0) and span >= math.log(SPAN_MIN_RATIO)
            if not admissible:
                print(f"[apex-nadir] ALLOC fit rejected: b={float(b):.3f}, "
                      f"T span x{math.exp(span):.1f} over {len(pts)} points "
                      f"(need 0<b<=1 and span >= x{SPAN_MIN_RATIO}); keeping previous model")
                # Still bank the points and the new maximum: the next input may
                # widen the span enough to make the fit identifiable.
                self.alloc.t_max_seen = max([self.alloc.t_max_seen] + [t for t, _ in pts])
                return False
            self.alloc.log_a, self.alloc.b = float(log_a), float(b)
        self.alloc.fitted = True
        self.alloc.n_points = len(pts)
        self.alloc.t_max_seen = max([self.alloc.t_max_seen] + [t for t, _ in pts])
        return True

    def allocation(self, total_tokens: int) -> int:
        """ALLOC(T): tokens one expert should receive for an input of T tokens."""
        return max(1, min(int(total_tokens), int(round(self.alloc.predict(total_tokens)))))

    def experts_for(self, total_tokens: int) -> int:
        """k = T / ALLOC(T). Expert count FOLLOWS the token budget instead of
        fighting it, so no fragment is ever too small to use and nothing is ever
        skipped. As experts mature ALLOC rises and k falls — K->0 becomes a
        measurable consequence rather than an artefact of the gate's entropy."""
        alloc = self.allocation(total_tokens)
        return max(1, min(int(configs.X_MAX), int(max(1, int(total_tokens) // max(1, alloc)))))

    # ── refresh ────────────────────────────────────────────────────────────
    def needs_refresh(self, total_tokens: int) -> bool:
        """True when this input is bigger than anything calibrated — the one case
        where ALLOC would be extrapolating instead of interpolating."""
        return int(total_tokens) > int(self.alloc.t_max_seen)

    def refresh_expert_ids(self, tkl_scores: Dict[int, float],
                           resident: Optional[Iterable[int]] = None) -> List[int]:
        """The 2*sqrt(E) experts to re-probe on a new-maximum input: the sqrt(E)
        best and sqrt(E) worst by CURRENT TKL. Using live TKL keeps the envelopes
        tracking who is actually best now, and costs 2*sqrt(E)*3 probes instead of
        re-running the whole pool.

        RANKED WITHIN THE RESIDENT SET when one is given. The caller can only
        probe experts that are loaded, and the sqrt(E) WORST by TKL are — by
        definition — the ones nothing has been routing to, so they are almost
        never resident. Ranking over the whole pool therefore returned a list the
        caller had to discard, `len(loaded) < 2` fired, and the refresh returned
        before probing anything. Measured: 0/4, 0/4, 0/2 experts on three
        consecutive live batches, which is why ALLOC(T) was never fitted and
        R_out sat at FRAGMENT_MIN for every expert in the pool.

        Best-and-worst-among-resident preserves the spread the envelopes need
        (the sample still brackets the pool) while returning experts that can
        actually be probed. Residency biases which experts get RE-CALIBRATED, not
        which get selected or scored, so it costs accuracy on cold experts'
        curves rather than corrupting the routing signal."""
        pool = dict(tkl_scores)
        if resident is not None:
            live = set(int(e) for e in resident)
            in_res = {e: v for e, v in pool.items() if int(e) in live}
            # Only fall back to the full pool if residency leaves us unable to
            # bracket anything; two points is the minimum for a spread.
            if len(in_res) >= 2:
                pool = in_res
        scored = sorted(pool.items(), key=lambda kv: kv[1], reverse=True)
        if not scored:
            return []
        k = max(1, int(math.isqrt(len(scored))))
        picked = [eid for eid, _ in scored[:k]] + [eid for eid, _ in scored[-k:]]
        return list(dict.fromkeys(picked))   # dedupe: best and worst overlap when the pool is small
    def _init_default_curves(self):
        for i in range(configs.EXPERT_POOL_SIZE):
            self.expert_curves[i] = ExpertCurves()
    def _eval_poly(self, coeffs: np.ndarray, x: float) -> float:
        return float(sum(c * (x ** i) for i, c in enumerate(coeffs)))
    def _eval_apex(self, expert_id: int, token_count: float) -> float:
        curves = self.expert_curves[expert_id]
        return max(0.0, self._eval_poly(curves.apex_coeffs, token_count))
    def _eval_nadir(self, expert_id: int, token_count: float) -> float:
        curves = self.expert_curves[expert_id]
        return max(0.0, self._eval_poly(curves.nadir_coeffs, token_count))
    def _eval_latency(self, expert_id: int, token_count: float) -> float:
        curves = self.expert_curves[expert_id]
        return max(1e-6, self._eval_poly(curves.latency_coeffs, token_count))
    def compute_r_out(self, expert_id: int, total_tokens: Optional[int] = None) -> float:
        """Tokens this expert should receive.

        Prefers ALLOC(T) from the fitted apex-nadir convolution whenever the
        caller knows the input size. The legacy path below (argmax of
        apex(t)/latency(t)) is kept only as an uncalibrated fallback, and it is
        DEGENERATE BY CONSTRUCTION: latency is linear through the origin while
        apex grows sublinearly, so the ratio decreases monotonically and its
        argmax is always the search floor. That is why every expert reported
        R_out == FRAGMENT_MIN == 32 forever, which in turn made the old
        below-nadir skip impossible to calibrate away. Efficiency ratios always
        favour the minimum; the convolution replaces the ratio with a peak
        between two fitted envelopes."""
        if self.alloc.fitted and total_tokens is not None:
            return float(self.allocation(total_tokens))
        curves = self.expert_curves[expert_id]
        if curves.r_out_cached is not None:
            return curves.r_out_cached
        floor = max(configs.FRAGMENT_MIN, int(curves.nadir_floor))
        ceiling = max(floor + 1, int(curves.apex_ceiling))
        best_t = floor
        best_ratio = -1.0
        step = max(1, (ceiling - floor) // 100)
        for t in range(floor, ceiling + 1, step):
            s_c = self._eval_apex(expert_id, float(t))
            c_e = self._eval_latency(expert_id, float(t))
            ratio = s_c / c_e
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = t
        search_lo = max(floor, best_t - step)
        search_hi = min(ceiling, best_t + step)
        for t in range(search_lo, search_hi + 1):
            s_c = self._eval_apex(expert_id, float(t))
            c_e = self._eval_latency(expert_id, float(t))
            ratio = s_c / c_e
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = t
        r_out = float(max(configs.FRAGMENT_MIN, best_t))
        curves.r_out_cached = r_out
        return r_out
    def compute_r_out_mean(self, expert_ids: List[int]) -> float:
        if not expert_ids:
            return float(configs.MAX_SEQ_LEN) / configs.K_DEFAULT
        return float(np.mean([self.compute_r_out(eid) for eid in expert_ids]))
    def has_calibration(self, expert_id: int) -> bool:
        """True once this expert's curves carry real data (calibration or live
        latency). False = cold; callers should use EXPERT_BOOTSTRAP_TOKENS."""
        return self.expert_curves[expert_id].has_data
    def r_out_or_bootstrap(self, expert_id: int) -> float:
        """INPUT fragment size for this expert: its convolution R_out once the
        curves have data, otherwise the cold-start bootstrap size. Governs how many
        tokens the expert READS (gathers context from)."""
        if self.expert_curves[expert_id].has_data:
            return self.compute_r_out(expert_id)
        return float(configs.EXPERT_BOOTSTRAP_TOKENS)
    def generation_length(self, expert_id: int, total_tokens: Optional[int] = None) -> int:
        """OUTPUT length for this expert's generated analysis: R_out once calibrated,
        else the EXPERT_GEN_MAX_TOKENS safety valve. Governed SEPARATELY from the
        input fragment — reading a big context to calibrate (bootstrap) does not mean
        writing a long analysis. Central truncates over-long analyses anyway."""
        if self.alloc.fitted and total_tokens is not None:
            return max(1, int(self.allocation(total_tokens)))
        if self.expert_curves[expert_id].has_data:
            return max(1, int(self.compute_r_out(expert_id, total_tokens)))
        return int(configs.EXPERT_GEN_MAX_TOKENS)
    @staticmethod
    def _fit_latency_coeffs(stats: np.ndarray) -> Optional[np.ndarray]:
        """Least-squares [c0, c1] for cost = c0 + c1*t from sufficient statistics.

        Returns None while the intercept is not identifiable — fewer than two
        probes, or every probe at the same token count (zero variance in t, so
        the normal equations are singular and any intercept fits equally well).
        The caller keeps the old proportional-cost estimate in that case."""
        n, st, st2, sw, stw = (float(v) for v in stats)
        if n < 2.0:
            return None
        denom = n * st2 - st * st
        if denom <= 1e-9:            # all probes at one size: c0 unidentifiable
            return None
        c1 = (n * stw - st * sw) / denom
        c0 = (sw - c1 * st) / n
        if c1 <= 0.0:                # cost must rise with tokens; a flat or
            return None              # falling fit is noise, not a cost model
        if c0 < 0.0:
            # A negative fixed cost is unphysical. Clamp to zero and refit the
            # slope through the origin so the two coefficients stay consistent.
            if st <= 1e-9:
                return None
            return np.array([0.0, sw / st], dtype=np.float64)
        return np.array([c0, c1], dtype=np.float64)

    def update_latency(self, expert_id: int, token_count: int, wall_time: float):
        """Accumulate one cost measurement and refit latency as c0 + c1*t.

        THE INTERCEPT IS THE POINT. Every write site used to store [0.0, slope]:
        cost strictly proportional to tokens, no fixed overhead. That single
        pinned zero is what made compute_r_out degenerate. With
        apex(t) = a0 + a1*t + a2*t^2 over latency c1*t, the ratio is

            a0/(c1*t) + a1/c1 + (a2/c1)*t

        whose derivative is -a0/(c1*t^2) + a2/c1. A fitted apex has a0 >= 0 and
        (where it is concave) a2 <= 0, so that derivative is never positive: the
        ratio falls monotonically and its argmax is ALWAYS the search floor.
        Measured on the stored curves: 0 of 100 experts had an interior peak and
        compute_r_out returned exactly FRAGMENT_MIN for the entire pool, which
        left get_distance_to_peak with zero variance across experts and made the
        never-trained route_head the whole of the ranking.

        There genuinely IS a fixed cost — adapter load, forward-pass setup — and
        it is precisely what makes a tiny fragment inefficient, which is the
        whole reason a goldilocks point exists at all. Fitting it restores the
        interior peak, and because c0 and c1 are then both expert-specific, R_out
        becomes per-expert instead of a pool-wide constant."""
        t = float(max(token_count, 1))
        w = float(wall_time)
        curves = self.expert_curves[expert_id]
        if not np.isfinite(w) or w <= 0.0:
            # An unrecorded wall time (record_probe's default 0.0) carries no
            # cost information. Folding it in would drag the fit toward a free
            # expert — the same fabricated-measurement failure as quality 0.
            return
        curves.lat_stats = curves.lat_stats + np.array([1.0, t, t * t, w, t * w], dtype=np.float64)
        fitted = self._fit_latency_coeffs(curves.lat_stats)
        if fitted is not None:
            curves.latency_coeffs = fitted
        else:
            # Intercept not identifiable yet — keep the previous behaviour (EMA
            # on a through-origin rate) so a single probe size degrades to
            # exactly what it did before rather than to something unfitted.
            old_slope = curves.latency_coeffs[1] if len(curves.latency_coeffs) > 1 else 0.001
            new_slope = configs.EMA_DECAY * old_slope + (1.0 - configs.EMA_DECAY) * (w / t)
            curves.latency_coeffs = np.array([0.0, new_slope], dtype=np.float64)
        curves.r_out_cached = None
        curves.has_data = True   # a real latency measurement is real data
    def check_monopoly_ceiling(self, expert_id: int, current_allocation: int) -> bool:
        curves = self.expert_curves[expert_id]
        return current_allocation > curves.apex_ceiling * configs.MONOPOLY_THRESHOLD
    def check_nadir_floor(self, expert_id: int, fragment_size: int) -> bool:
        """INFORMATIONAL ONLY: is this fragment below the expert's underfit floor?

        This must NEVER gate execution. The nadir is an ALLOCATION signal — when
        tokens are scarce we run FEWER experts (k = T/ALLOC(T)), we never drop an
        expert. Callers previously treated a True here as 'skip this expert',
        which silently disabled the entire pool for any question shorter than
        FRAGMENT_MIN*k and made Timeline B identical to Timeline A. Use it for
        telemetry; use experts_for() to decide how many experts to run."""
        curves = self.expert_curves[expert_id]
        return fragment_size < curves.nadir_floor
    def get_distance_to_peak(self, expert_id: int, current_allocation: int) -> float:
        r_out = self.compute_r_out(expert_id)
        if r_out < 1e-6:
            return float('inf')
        return abs(current_allocation - r_out) / r_out
    def fit_curves_from_calibration(self, expert_id: int, calibration_data: dict):
        curves = self.expert_curves[expert_id]
        if "token_counts" in calibration_data and "quality_scores" in calibration_data:
            tc = np.array(calibration_data["token_counts"], dtype=np.float64)
            qs = np.array(calibration_data["quality_scores"], dtype=np.float64)
            if len(tc) >= 3:
                X = np.vstack([np.ones_like(tc), tc, tc ** 2]).T
                coeffs, _, _, _ = np.linalg.lstsq(X, qs, rcond=None)
                curves.apex_coeffs = coeffs
                if coeffs[2] < 0:
                    curves.apex_ceiling = max(float(configs.FRAGMENT_MIN), float(-coeffs[1] / (2 * coeffs[2])))
                else:
                    curves.apex_ceiling = float(tc.max())
        if "gradient_coherence" in calibration_data and "token_counts" in calibration_data:
            gc = np.array(calibration_data["gradient_coherence"], dtype=np.float64)
            tc = np.array(calibration_data["token_counts"], dtype=np.float64)
            valid = tc[gc > 0.1]
            if len(valid) > 0:
                curves.nadir_floor = max(float(configs.FRAGMENT_MIN), float(valid.min()))
            curves.nadir_coeffs = np.array([curves.nadir_floor, 0.0])
        if "wall_times" in calibration_data and "token_counts" in calibration_data:
            wt = np.array(calibration_data["wall_times"], dtype=np.float64)
            tc = np.array(calibration_data["token_counts"], dtype=np.float64)
            if len(tc) > 0:
                # Same two-parameter fit as the live path, so a calibrated expert
                # and a probe-warmed one produce comparable curves.
                curves.lat_stats = np.array([
                    float(len(tc)), float(tc.sum()), float((tc * tc).sum()),
                    float(wt.sum()), float((tc * wt).sum()),
                ], dtype=np.float64)
                fitted = self._fit_latency_coeffs(curves.lat_stats)
                curves.latency_coeffs = (
                    fitted if fitted is not None
                    else np.array([0.0, float(np.mean(wt / np.maximum(tc, 1)))], dtype=np.float64)
                )
        curves.r_out_cached = None
        curves.has_data = True   # curves fitted from calibration → no longer cold
    def reset_r_t_curve(self, expert_id: int):
        # Called on lateral migration. The latency curve is domain/hardware-bound,
        # so it resets and the expert goes cold (has_data=False) → it re-bootstraps
        # and re-convolves in the new domain. The structural apex/nadir curves are
        # left intact: they describe the expert's own capacity and travel with it.
        curves = self.expert_curves[expert_id]
        curves.latency_coeffs = np.array([0.0, 0.001])
        # The statistics go with the curve: they were accumulated in the old
        # domain, and keeping them would let stale cost evidence outvote every
        # measurement taken after the move.
        curves.lat_stats = np.zeros(5, dtype=np.float64)
        curves.r_out_cached = None
        curves.has_data = False
    def save(self):
        from pathlib import Path
        Path(self.calibration_path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for eid, curves in self.expert_curves.items():
            data[f"apex_{eid}"] = curves.apex_coeffs
            data[f"nadir_{eid}"] = curves.nadir_coeffs
            data[f"latency_{eid}"] = curves.latency_coeffs
            data[f"latstats_{eid}"] = curves.lat_stats
            data[f"meta_{eid}"] = np.array([curves.apex_ceiling, curves.nadir_floor])
        # ALLOC(T) — the fitted apex-nadir convolution, plus the goldilocks points
        # it was fitted from so a restart can extend rather than re-derive it.
        data["central_min_cap"] = np.array([float(self.central_min_cap)], dtype=np.float64)
        data["alloc"] = np.array([
            self.alloc.log_a, self.alloc.b, float(self.alloc.fitted),
            float(self.alloc.t_max_seen), float(self.alloc.n_points),
        ], dtype=np.float64)
        if self.goldilocks_points:
            data["goldilocks"] = np.array(self.goldilocks_points, dtype=np.float64)
        np.savez(self.calibration_path, **data)
    def save_latency_store(self):
        from pathlib import Path
        Path(self.latency_store_path).parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for eid, curves in self.expert_curves.items():
            data[f"latency_{eid}"] = curves.latency_coeffs
            # Without the statistics the fit restarts from n=0 every session and
            # the intercept can never re-identify itself across restarts.
            data[f"latstats_{eid}"] = curves.lat_stats
        np.savez(self.latency_store_path, **data)
    def load(self):
        try:
            data = np.load(self.calibration_path, allow_pickle=False)
            for eid in range(configs.EXPERT_POOL_SIZE):
                curves = self.expert_curves[eid]
                if f"apex_{eid}" in data:
                    curves.apex_coeffs = data[f"apex_{eid}"]
                if f"nadir_{eid}" in data:
                    curves.nadir_coeffs = data[f"nadir_{eid}"]
                if f"latency_{eid}" in data:
                    curves.latency_coeffs = data[f"latency_{eid}"]
                if f"latstats_{eid}" in data:
                    curves.lat_stats = np.asarray(data[f"latstats_{eid}"], dtype=np.float64)
                if f"meta_{eid}" in data:
                    meta = data[f"meta_{eid}"]
                    curves.apex_ceiling = float(meta[0])
                    curves.nadir_floor = float(meta[1])
                if f"apex_{eid}" in data or f"latency_{eid}" in data:
                    curves.has_data = True   # persisted calibration → warm at boot
                curves.r_out_cached = None
            if "central_min_cap" in data:
                self.central_min_cap = float(data["central_min_cap"][0])
            if "alloc" in data:
                a = data["alloc"]
                loaded = AllocationModel(
                    log_a=float(a[0]), b=float(a[1]), fitted=bool(a[2]),
                    t_max_seen=int(a[3]), n_points=int(a[4]),
                )
                # Same admissibility test as fit_allocation, applied on LOAD.
                # A model that was inadmissible when fitted is equally
                # inadmissible when restored, and a persisted b <= 0 would be
                # trusted for the whole next session — allocation shrinking as
                # input grows, ALLOC(512)=1, experts_for pinned at the ceiling.
                # Validate at the boundary, not only at the point of creation.
                if loaded.fitted and not (0.0 < loaded.b <= 1.0):
                    print(f"[apex-nadir] persisted ALLOC rejected on load: b={loaded.b:.3f} "
                          f"(need 0<b<=1); keeping t_max_seen={loaded.t_max_seen}, refitting from probes")
                    loaded.fitted = False
                    loaded.log_a, loaded.b = 0.0, 1.0
                self.alloc = loaded
            if "goldilocks" in data:
                self.goldilocks_points = [
                    (int(row[0]), float(row[1]), float(row[2])) for row in data["goldilocks"]
                ]
        except FileNotFoundError:
            pass
        try:
            data = np.load(self.latency_store_path, allow_pickle=False)
            for eid in range(configs.EXPERT_POOL_SIZE):
                if f"latency_{eid}" in data:
                    self.expert_curves[eid].latency_coeffs = data[f"latency_{eid}"]
                    self.expert_curves[eid].r_out_cached = None
                if f"latstats_{eid}" in data:
                    self.expert_curves[eid].lat_stats = np.asarray(
                        data[f"latstats_{eid}"], dtype=np.float64)
        except FileNotFoundError:
            pass
