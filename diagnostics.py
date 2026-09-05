from __future__ import annotations

import os
import re
import subprocess
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import mlx.core as mx

import configs


@dataclass
class SystemSnapshot:
    batch_index: int
    tokens_processed: int
    time_in_bound: float
    thermal_state: float
    ram_headroom_mb: float
    ssd_read_rate_mb: float
    k_used: int
    x_used: int


TEMP_SCALE = 100.0   # the inverted scale's top: 1 C maps to 100


def temp_divisor(celsius: float) -> float:
    """STANDARD convention, used wherever temperature DIVIDES.
    Higher = hotter = larger divisor = smaller result. Hot suppresses."""
    return max(1e-6, float(celsius))


def temp_multiplier(celsius: float) -> float:
    """INVERTED convention, used wherever temperature MULTIPLIES.
    The scale is flipped so 1 C -> 100: cool is a large multiplier, hot a small
    one. The same physical reading therefore pushes in opposite directions
    depending on which side of the expression it lands on — which is the point.
    Using the raw value in a multiplication would make a hot machine score
    HIGHER, silently inverting the whole force."""
    return max(0.0, TEMP_SCALE + 1.0 - float(celsius))


def d_temp_multiplier(d_celsius: float) -> float:
    """Change in temperature, on the inverted scale. Cooling (dT < 0) becomes a
    POSITIVE contribution, heating a negative one, because a falling temperature
    is a rising coolness."""
    return -float(d_celsius)


def expert_demand(
    experts: int,
    tokens_left: int,
    temperature: float,
    time_elapsed: float,
    d_temp: float,
    rate_of_process: float,
    tokens_processed: int,
) -> float:
    """The demand expression:

        (E / (temperature · time_elapsed))
            × tokens_left × Δtemperature(inverted)
            × rate_of_process × tokens_processed

    TOKENS_LEFT MULTIPLIES, it does not divide. Work remaining is a reason to
    spread wider, not to ration: more tokens means each expert still gets a share
    at or above its goldilocks allocation, so the marginal expert keeps earning
    its heat and load. In the denominator it produced the opposite force and
    contradicted the allocation-fit term in evaluate_configuration, which had the
    same input pushing up.

    Temperature appears TWICE under opposite conventions — raw in the division,
    inverted in the multiplication — so hot both shrinks the base and fails to
    earn a multiplier, while cool does the reverse. What DIVIDES is what has
    already been consumed (heat, elapsed time); what MULTIPLIES is what is still
    to be gained (work outstanding, throughput, cooling)."""
    denom = temp_divisor(temperature) * max(1e-6, float(time_elapsed))
    base = float(experts) / denom
    return float(base
                 * max(0.0, float(tokens_left))
                 * d_temp_multiplier(d_temp)
                 * max(0.0, float(rate_of_process))
                 * max(0.0, float(tokens_processed)))


def norm_demand(
    experts: int,
    e_max: int,
    tokens_left: int,
    tokens_total: int,
    temperature: float,
    d_temp: float,
    rate_of_process: float,
    best_rate: float,
    tokens_processed: int,
    time_elapsed: float,
    time_budget: Optional[float] = None,
    d_temp_range: float = 10.0,
) -> Dict[str, float]:
    """The demand expression with EVERY factor normalised to [0, 1], so the
    product is in [0, 1] too and nothing needs a scale afterwards.

    Normalising each factor rather than the final product is what keeps this
    combinable: the raw expression ran to ~1.8e7, which cannot be multiplied
    against terms that live in [0,1] without erasing them.

    It also removes the sign problem. d_temp mapped to [0,1] with 0.5 as steady
    means heating is a SMALL POSITIVE rather than a negative — a weak demand
    instead of an inverted one, so nothing downstream silently flips.

    Every factor is a fraction of something measured, so all of them read the
    same way: 1.0 is "as favourable as this can get", 0.0 is "no case at all".

        f_experts    x / E_max              how much of the pool this uses
        f_left       tokens_left / total    work still to be gained
        f_cool       inverted temp / SCALE  1 at cold, 0 at the ceiling
        f_dtemp      0.5 - dT/(2*range)     >0.5 cooling, <0.5 heating
        f_rate       rate / best_rate       throughput against the best seen
        f_done       processed / total      progress already banked
        f_time       budget left, or decay  time consumed counts against
    """
    e_max = max(1, int(e_max))
    tokens_total = max(1, int(tokens_total))
    f_experts = max(0.0, min(1.0, float(experts) / e_max))
    f_left = max(0.0, min(1.0, float(tokens_left) / tokens_total))
    f_cool = max(0.0, min(1.0, temp_multiplier(temperature) / TEMP_SCALE))
    rng = max(1e-6, float(d_temp_range))
    f_dtemp = max(0.0, min(1.0, 0.5 - float(d_temp) / (2.0 * rng)))
    f_rate = max(0.0, min(1.0, float(rate_of_process) / max(1e-6, float(best_rate))))
    f_done = max(0.0, min(1.0, float(tokens_processed) / tokens_total))
    if time_budget and time_budget > 0:
        f_time = max(0.0, min(1.0, 1.0 - float(time_elapsed) / float(time_budget)))
    else:
        f_time = 1.0 / (1.0 + max(0.0, float(time_elapsed)))
    demand = (f_experts * f_left * f_cool * f_dtemp * f_rate * f_done * f_time)
    return {
        "demand": float(demand), "f_experts": f_experts, "f_left": f_left,
        "f_cool": f_cool, "f_dtemp": f_dtemp, "f_rate": f_rate,
        "f_done": f_done, "f_time": f_time,
    }


def select_k_cap(k_max: int, ranked_tkl: List[float], **conditions) -> Dict[str, object]:
    """Evaluate every candidate x in 1..K and return the one that scores best —
    the new K cap. Called at the moment an expert swap is due, since that is when
    changing the count is free.

    The per-x function must be evaluate_configuration, NOT norm_demand.
    norm_demand's only x-dependent factor is f_experts = x/E_max; the other six
    describe the SITUATION and are identical for every candidate. A product with
    one increasing factor and six constants is monotonically increasing, so its
    argmax is always K and there is no choice being made. The interior optimum
    lives entirely on the cost side — thermal rising with x, RAM feasibility,
    allocation share falling as T/x, load time per expert — which is what
    evaluate_configuration models.

    K itself is not fixed: it is the hard cap from RAM, so freeing memory raises
    the ceiling and this search simply gets more candidates to consider."""
    k_max = max(1, int(k_max))
    scores = {x: evaluate_configuration(x, ranked_tkl, **conditions)
              for x in range(1, k_max + 1)}
    best = max(scores, key=scores.get)
    return {"k_cap": int(best), "scores": scores,
            "feasible": [x for x, v in scores.items() if v > 0.0]}


def evaluate_configuration(
    x: int,
    ranked_tkl: List[float],
    thermal: float,
    ram_free_mb: float,
    sec_per_expert: float,
    total_tokens: Optional[int] = None,
    alloc_target: Optional[float] = None,
    deadline_s: Optional[float] = None,
    swaps: Optional[int] = None,
    resident: int = 0,
    load_time_s: float = 0.0,
    time_exponent: float = 2.0,
    d_temp: float = 0.0,
    d_tokens: float = 0.0,
    d_time: float = 0.0,
    tokens_left: Optional[int] = None,
    time_elapsed: float = 0.0,
    x_current: int = 1,
    ranked_ids: Optional[List[int]] = None,
    resident_ids: Optional[Iterable[int]] = None,
) -> float:
    """THE evaluation function — what a candidate expert-count is worth.

        S(x) = value(x) · thermal(x) · ram(x) · time(x) · swap(x)

    value(x) = sum of the TOP-x experts' TKL, best first.
        This is where "best expert, then good experts, then losers last" becomes
        arithmetic. Because the pool is ranked, the x-th expert contributes only
        the x-th-best score, so value rises CONCAVELY while every cost rises at
        least linearly. That single fact is what makes an interior optimum exist
        without tuning anything: at some x the next expert stops paying for its
        own heat, memory and load time.

    thermal(x) headroom below the throttle point, using the temperature this
        configuration is predicted to reach — more experts, more heat.
    ram(x)     hard feasibility. Zero if the configuration cannot fit; there is
        no partial credit for nearly fitting.
    time(x)    "need to answer fast". With a deadline this is how comfortably x
        finishes inside it; without one, plain speed.
    swap(x)    penalty for loads beyond the unavoidable one-per-expert — which is
        exactly what schedule_by_expert drives toward, so a schedule that keeps
        the heaviest experts resident scores higher here.

    Multiplied, not summed, so a configuration that is infeasible on any single
    axis scores ~0 rather than averaging its way to respectability."""
    if x < 1 or not ranked_tkl:
        return 0.0
    value = float(sum(sorted(ranked_tkl, reverse=True)[:x]))
    # ALLOCATION FIT — why more tokens justify more experts.
    # Splitting T tokens across x experts gives each T/x. If that share falls
    # below the goldilocks allocation the expert is starved and contributes a
    # fraction of what it could, so its TKL is discounted accordingly. With few
    # tokens, six experts each get a starved sliver and six is worth less than
    # two; with many tokens the share stays healthy and the sixth expert is worth
    # having. That is the whole "tokens are more -> more experts" effect, and it
    # comes straight out of apex-nadir rather than being asserted.
    if total_tokens and alloc_target and alloc_target > 0:
        share = float(total_tokens) / float(x)
        value *= max(0.0, min(1.0, share / float(alloc_target)))
    throttle = max(1.0, float(configs.THERMAL_THROTTLE_TEMP))
    # THERMAL — level AND trend. A machine at 70C and cooling has far more
    # headroom than one at 70C and climbing, and a snapshot cannot tell them
    # apart. d_temp gives the trajectory; it also gives the per-expert heat slope
    # for free (d_temp / x_current), replacing a hardcoded 0.12 with something
    # this machine actually measured under this load.
    per_expert_heat = (float(d_temp) / max(1, int(x_current))) if d_temp else 0.0
    predicted_t = float(thermal) + per_expert_heat * (x - max(1, int(x_current)))
    predicted_t += float(d_temp)          # the trend continues for at least one more step
    thermal_term = max(0.0, min(1.0, 1.0 - predicted_t / throttle))
    need_mb = float(configs.EXPERT_RAM_MB) * x
    ram_term = 1.0 if ram_free_mb >= need_mb else 0.0
    # TIME — an UPWARD force, even though experts run sequentially.
    # Attention is quadratic in sequence length: one expert on T tokens costs
    # ~O(T^2), while x experts each on T/x cost x*O((T/x)^2) = O(T^2/x). So
    # splitting work across more experts REDUCES total compute. Generally, if
    # per-expert cost scales as share^p then total = sec_full / x^(p-1), which
    # falls with x for any p > 1. Modelling it as sec_per_expert * x assumed the
    # per-expert cost was independent of its share, which inverted the sign of
    # the whole force and made "answer faster" argue for fewer experts.
    # Loading is the counterweight: every expert costs one load, linear in x.
    p = max(1.0, float(time_exponent))
    compute_s = max(1e-6, float(sec_per_expert)) / (float(x) ** (p - 1.0))
    est_time = compute_s + float(load_time_s) * x
    time_term = (max(0.0, min(1.0, deadline_s / est_time)) if deadline_s
                 else 1.0 / (1.0 + est_time))
    # URGENCY — remaining work against remaining budget.
    # Instantaneous speed says nothing about whether the job will land. Project
    # the work left at the throughput being achieved: if that overruns the budget
    # already partly spent, being behind schedule is itself an upward force,
    # because the fix for falling behind is to split the remaining tokens wider.
    if tokens_left and d_tokens > 0 and d_time > 0:
        throughput = (float(d_tokens) / float(d_time)) * (float(x) ** (p - 1.0))
        projected_s = float(tokens_left) / max(1e-6, throughput)
        if deadline_s:
            budget_left = max(1e-6, float(deadline_s) - float(time_elapsed))
            behind = projected_s / budget_left        # > 1 = will overrun
            time_term *= max(0.0, min(1.0, 1.0 / behind)) if behind > 1.0 else 1.0
        else:
            time_term *= 1.0 / (1.0 + projected_s / max(1e-6, float(d_time)))
    # SWAPS — also an UPWARD force, via RETENTION.
    # Experts already resident cost nothing to reuse; only going beyond the
    # resident set buys new loads. Counting swaps as always >= x saw the cost of
    # loading and missed the benefit of keeping experts in place, so it pushed
    # down when it should push up. With the same experts recurring — which stable
    # domain assignment guarantees — using what is already loaded is free.
    # No falsy guard: resident=0 means NOTHING is loaded, so all x must be
    # fetched — the worst case. Treating 0 as "unknown, no penalty" gave a cold
    # cache the same score as a fully warm one.
    #
    # Residency is a SET, not a count. `x - resident` assumes the resident
    # experts are the top-ranked ones, which is precisely the assumption a swap
    # decision exists because it cannot make: with 6 resident and none of them in
    # the top 3, x=3 scored max(0, 3-6) = 0 new loads — free — when all three
    # actually have to be fetched. Given the ids, count the ones genuinely
    # missing; fall back to the count-based estimate only when ids are absent.
    if ranked_ids is not None and resident_ids is not None:
        live = set(int(e) for e in resident_ids)
        top_x = [int(e) for e in list(ranked_ids)[:int(x)]]
        new_loads = sum(1 for e in top_x if e not in live)
    else:
        new_loads = max(0, int(x) - int(resident))
    sw = int(swaps) if swaps is not None else new_loads
    swap_term = 1.0 / (1.0 + max(0, sw))
    return float(value * thermal_term * ram_term * time_term * swap_term)


class Diagnostics:
    def __init__(self):
        self.history: List[SystemSnapshot] = []
        self.x_next: int = configs.X_MAX
        self._batch_index: int = 0
        self._thermal_estimate: float = 58.0
        self._thermal_is_exact: bool = False
        # ── live memory governor: OBSERVED peak-utilization based ──
        # Extrapolated cost models underestimate the generation spike and overshoot
        # into OOM, so control is on the actual measured peak vs usable RAM instead.
        self._mem_base_mb: float = 0.0       # resident central+gate, measured
        self._mem_usable_mb: float = 0.0     # ceiling the MLX peak may reach (base+free), live
        self._last_peak_mb: float = 0.0      # last batch's observed MLX high-water mark
        # Concurrency is chosen at swap time by select_k_cap, which evaluates
        # every candidate 1..K against the live conditions.
        self._last_swaps: int = 0
        self._load_time_s: float = 0.0
        self._hard_cap: Optional[int] = None
        self._inputs_since_cap: int = 0

    def set_memory_baseline(self, base_mb: float, usable_mb: float) -> None:
        """Record measured resident base (central+gate) and the usable ceiling
        (base + currently-available RAM). Peak utilization is measured live after."""
        self._mem_base_mb = base_mb
        self._mem_usable_mb = usable_mb
        self._last_peak_mb = 0.0

    def set_usable(self, usable_mb: float) -> None:
        """Refresh the usable-RAM ceiling from a LIVE reading so it tracks RAM other
        consumers (e.g. HF data-stream buffers) take after boot."""
        if usable_mb > 0.0:
            self._mem_usable_mb = usable_mb

    def observe_memory(self, x_used: int) -> None:
        """Record the batch's real MLX peak (the OOM-relevant high-water mark) and
        reset it. Control is on this observed peak vs usable — no extrapolation."""
        from splitter import get_peak_memory_mb, reset_peak_memory
        peak = get_peak_memory_mb()
        reset_peak_memory()
        if peak > 0.0:
            self._last_peak_mb = peak

    def peak_util(self) -> float:
        """Last batch's peak memory as a fraction of usable RAM (0.0 if unmeasured)."""
        if self._mem_usable_mb <= 0.0 or self._last_peak_mb <= 0.0:
            return 0.0
        return self._last_peak_mb / self._mem_usable_mb

    def can_fit_expert(self) -> bool:
        """Whether to run at least one expert this batch. Only False when the last
        peak was genuinely near the ceiling (MEM_FALLBACK_FRAC) — running 1 expert in
        the 80–92% band is safe (it already ran there), so don't needlessly drop to
        Central-only; recommended_x just won't GROW past 1 there."""
        u = self.peak_util()
        return u == 0.0 or u < configs.MEM_FALLBACK_FRAC

    # ── system hard cap ────────────────────────────────────────────────────
    def _oom_marker_path(self):
        from pathlib import Path
        return Path(configs.CHECKPOINT_DIR).parent / "oom_marker.json"

    def record_oom(self, x_attempted: int) -> None:
        """Persist an OOM so the lesson OUTLIVES THE PROCESS.

        A Metal OOM aborts the process — nothing downstream runs, so no particle
        is written and no score is recorded. The crash therefore erases its own
        evidence: on restart the governor sees a map in which the fatal
        configuration never appears, and it climbs straight back into it. The
        survivors are the only things ever recorded, so the space near the cliff
        looks smooth right up to the edge.

        Writing a marker to DISK is what breaks that loop. A supervisor calls
        this on the restart following a crash (or the caller wraps the risky
        step), and _apply_oom_marker caps the system below the value that died —
        a tombstone that survives the thing that killed the process."""
        import json
        try:
            p = self._oom_marker_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            hist = []
            if p.exists():
                hist = json.loads(p.read_text()).get("attempts", [])
            hist.append(int(x_attempted))
            p.write_text(json.dumps({"attempts": hist[-16:]}))
        except Exception as e:
            print(f"[warn] could not persist OOM marker: {e}")
        self._hard_cap = max(configs.X_MIN, int(x_attempted) - 1)
        print(f"[oom] x={x_attempted} recorded; hard cap -> {self._hard_cap}")

    def _inflight_path(self):
        from pathlib import Path
        return Path(configs.CHECKPOINT_DIR).parent / "inflight_x.json"

    def arm_oom_watch(self, x_attempted: int) -> None:
        """Write "we are about to run at x" to disk, BEFORE the risky work.

        record_oom() can only be called by something that survives the crash, and
        a Metal OOM does not permit that: kIOGPUCommandBufferCallbackError is an
        uncatchable C++ abort, so no except block runs, no marker is written, and
        the fatal configuration never enters the record. On restart the governor
        sees a map in which nothing ever died and climbs straight back into it.

        The fix is to write the tombstone in advance and clear it on success. A
        marker still present at boot means the process died while running that x
        — which is exactly the evidence the crash would otherwise have erased."""
        try:
            import json
            p = self._inflight_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"x": int(x_attempted)}))
        except Exception:
            pass          # never let bookkeeping break the batch

    def disarm_oom_watch(self) -> None:
        """Batch survived — clear the tombstone."""
        try:
            p = self._inflight_path()
            if p.exists():
                p.unlink()
        except Exception:
            pass

    def claim_crashed_run(self) -> Optional[int]:
        """At boot: if a tombstone survived, the last run died at that x. Record
        it and clear it. Returns the x that died, or None for a clean start."""
        try:
            import json
            p = self._inflight_path()
            if not p.exists():
                return None
            x = int(json.loads(p.read_text()).get("x", 0))
            p.unlink()
            if x >= 1:
                print(f"[oom] previous run died at x={x} — recording tombstone")
                self.record_oom(x)
                return x
        except Exception as e:
            print(f"[warn] could not read in-flight marker: {e}")
        return None

    def _apply_oom_marker(self) -> Optional[int]:
        """Lowest x known to have died, minus one — read at boot."""
        import json
        try:
            p = self._oom_marker_path()
            if not p.exists():
                return None
            attempts = json.loads(p.read_text()).get("attempts", [])
            return max(configs.X_MIN, min(attempts) - 1) if attempts else None
        except Exception:
            return None

    def refresh_hard_cap(self, force: bool = False) -> int:
        """Re-derive the system hard cap from LIVE memory.

        The cap is not a constant computed once at boot: other processes take
        memory, the OS swaps, and the measured model costs can drift. Re-checking
        every E inputs keeps it honest, and `force=True` re-checks immediately —
        which is what a supervisor does after a crash, since an OOM is proof the
        previous cap was wrong.

        Any recorded OOM binds it further: the cap can never exceed one below the
        smallest x known to have died."""
        from splitter import experts_per_batch
        self._inputs_since_cap += 1
        every = max(1, int(configs.EXPERT_POOL_SIZE))
        if not force and self._hard_cap is not None and self._inputs_since_cap < every:
            return self._hard_cap
        self._inputs_since_cap = 0
        cap = experts_per_batch()
        oom = self._apply_oom_marker()
        if oom is not None:
            cap = min(cap, oom)
        prev = self._hard_cap
        self._hard_cap = max(configs.X_MIN, min(configs.X_MAX, cap))
        if prev is not None and prev != self._hard_cap:
            print(f"[cap] hard cap {prev} -> {self._hard_cap}"
                  + (f" (OOM-bound at {oom})" if oom is not None else ""))
        return self._hard_cap

    def memory_ceiling(self) -> int:
        """The hard cap the governor may never exceed: whatever the RAM formula
        allows, and never more than X_MAX.

        This used to return X_MAX alone, so the RAM-derived cap only seeded the
        starting value and never bounded the search — the CEM could climb to 6 on
        a machine whose memory supported 1. The cap is the physical limit; X_MAX
        is only a policy ceiling on top of it."""
        return self.refresh_hard_cap()

    def _time_pressure(self, recent_time: float) -> bool:
        """True if recent batch time is well above the run's fastest (unthrottled)
        batch — a RELATIVE, self-calibrating slowdown signal (replaces a hardcoded
        absolute-seconds wall that mis-fired on every 7B+generation batch)."""
        if len(self.history) < 3:
            return False
        times = [s.time_in_bound for s in self.history if s.time_in_bound > 0.0]
        if not times:
            return False
        return recent_time > 2.0 * min(times)

    def _recent_tps(self) -> Optional[float]:
        """Tokens/sec of the most recent batch (per-batch delta / its wall time)."""
        if not self.history:
            return None
        last = self.history[-1]
        prev = self.history[-2].tokens_processed if len(self.history) >= 2 else 0
        delta = last.tokens_processed - prev
        return delta / last.time_in_bound if last.time_in_bound > 0 and delta > 0 else None

    def recommended_x(self, x_used_last: int) -> int:
        """Experts to run next batch — controlled by OBSERVED peak-memory utilization
        (last measured peak / usable RAM), NOT an extrapolated cost model (which
        underestimated the generation spike and overshot into OOM). Also backs off on
        SoC throttle onset or a throughput collapse. Grow +1 only when the last peak
        left real headroom; hold in the mid-band; drop when it nears the ceiling.
        Empirical → cannot overshoot: one extra expert can't exceed the headroom the
        grow-threshold guarantees."""
        util = self.peak_util()
        thermal_hot = bool(self.history) and self.history[-1].thermal_state > configs.THERMAL_THROTTLE_TEMP * configs.THERMAL_BACKOFF_FRAC
        tps = self._recent_tps()
        if tps is not None:
            self._best_tps = max(getattr(self, "_best_tps", 0.0), tps)
        throughput_collapse = (
            tps is not None and getattr(self, "_best_tps", 0.0) > 0.0
            and tps < configs.THROUGHPUT_COLLAPSE_FRAC * self._best_tps
        )
        if util == 0.0:                                  # no measurement yet — ramp cautiously
            return max(configs.X_MIN, min(configs.X_MAX, x_used_last + 1))
        if thermal_hot or throughput_collapse or util > configs.MEM_BACKOFF_FRAC:
            return max(configs.X_MIN, x_used_last - 1)   # back off under real pressure
        if util < configs.MEM_GROW_HEADROOM_FRAC:
            return min(configs.X_MAX, x_used_last + 1)   # comfortable headroom → grow
        return x_used_last                               # mid-band → hold steady

    def update(self, tokens_processed: int, time_in_bound: float, x_used: int, k_used: int,
               ranked_tkl: Optional[List[float]] = None,
               total_tokens: Optional[int] = None,
               tokens_left: Optional[int] = None,
               alloc_target: Optional[float] = None,
               resident: int = 0,
               ranked_ids: Optional[List[int]] = None,
               resident_ids: Optional[Iterable[int]] = None) -> int:
        ram_headroom_mb = self._read_ram()
        ssd_read_rate_mb = self._read_ssd_rate()
        snap = SystemSnapshot(
            batch_index=self._batch_index,
            tokens_processed=tokens_processed,
            time_in_bound=time_in_bound,
            thermal_state=self._read_thermal(time_in_bound, x_used, k_used, ram_headroom_mb, ssd_read_rate_mb),
            ram_headroom_mb=ram_headroom_mb,
            ssd_read_rate_mb=ssd_read_rate_mb,
            k_used=k_used,
            x_used=x_used,
        )
        self.history.append(snap)
        self._batch_index += 1
        # SWAP MOMENT — this is when changing the expert count is free, so this is
        # when the cap is re-chosen: evaluate every candidate 1..K against the
        # live conditions and take the best. Without the pool's ranked scores
        # there is nothing to evaluate, so the current count simply stands.
        if not ranked_tkl:
            self.x_next = max(configs.X_MIN, min(self.memory_ceiling(), x_used))
            return self.x_next
        d_temp = 0.0
        if len(self.history) >= 2:
            d_temp = snap.thermal_state - self.history[-2].thermal_state
        result = select_k_cap(
            self.memory_ceiling(), list(ranked_tkl),
            thermal=snap.thermal_state, ram_free_mb=ram_headroom_mb,
            sec_per_expert=max(1e-3, time_in_bound / max(1, x_used)),
            total_tokens=total_tokens, alloc_target=alloc_target,
            tokens_left=tokens_left, time_elapsed=float(self._batch_index),
            resident=int(resident), load_time_s=self._load_time_s,
            d_temp=d_temp, x_current=int(x_used),
            # ids, so the swap term counts experts genuinely missing from the
            # resident set rather than assuming the top-x are the loaded ones.
            ranked_ids=ranked_ids,
            resident_ids=resident_ids,
        )
        self.x_next = int(result["k_cap"])
        return self.x_next

    def validate_thermal_regression(self) -> dict:
        if not self.history:
            return {
                "history_len": 0,
                "x_next": self.x_next,
                "bounded": configs.X_MIN <= self.x_next <= configs.X_MAX,
                "thermal_guard_active": False,
                "recent_avg_thermal": 0.0,
                "recent_avg_time_in_bound": 0.0,
                "thermal_source": "none",
            }
        latest = self.history[-1]
        recent = self.history[-3:]
        recent_avg_thermal = sum(snap.thermal_state for snap in recent) / len(recent)
        recent_avg_time = sum(snap.time_in_bound for snap in recent) / len(recent)
        thermal_guard_active = (
            latest.thermal_state > configs.THERMAL_THROTTLE_TEMP * 0.9
            or (
                not self._thermal_is_exact
                and len(self.history) >= 3
                and (
                    self._time_pressure(recent_avg_time)
                    or recent_avg_thermal > configs.THERMAL_THROTTLE_TEMP * 0.82
                )
            )
        )
        return {
            "history_len": len(self.history),
            "latest_batch_index": latest.batch_index,
            "x_used": latest.x_used,
            "x_next": self.x_next,
            "bounded": configs.X_MIN <= self.x_next <= configs.X_MAX,
            "thermal_guard_active": thermal_guard_active,
            "thermal": latest.thermal_state,
            "recent_avg_thermal": recent_avg_thermal,
            "recent_avg_time_in_bound": recent_avg_time,
            "ram_mb": latest.ram_headroom_mb,
            "ssd_read_rate_mb": latest.ssd_read_rate_mb,
            "thermal_source": "direct" if self._thermal_is_exact else "proxy_or_estimate",
        }

    def _read_thermal(
        self,
        time_in_bound: float,
        x_used: int,
        k_used: int,
        ram_headroom_mb: float,
        ssd_read_rate_mb: float,
    ) -> float:
        direct = self._read_powermetrics_thermal()
        if direct is not None:
            self._thermal_estimate = direct
            self._thermal_is_exact = True
            return direct
        proxy = self._read_pmset_thermal_proxy()
        if proxy is not None:
            self._thermal_estimate = proxy
            self._thermal_is_exact = False
            return proxy
        self._thermal_is_exact = False
        return self._estimate_thermal(time_in_bound, x_used, k_used, ram_headroom_mb, ssd_read_rate_mb)

    def _read_powermetrics_thermal(self) -> Optional[float]:
        commands = []
        if os.geteuid() == 0:
            commands.append(["powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"])
        else:
            commands.append(["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"])
            commands.append(["powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"])
        for command in commands:
            try:
                out = subprocess.check_output(
                    command,
                    text=True,
                    timeout=2,
                    stderr=subprocess.DEVNULL,
                )
                for line in out.splitlines():
                    if "CPU die temperature" in line:
                        return float(line.split(":")[1].strip().split()[0])
            except Exception:
                continue
        return None

    def _read_pmset_thermal_proxy(self) -> Optional[float]:
        # thermlog is deliberately excluded — it blocks for the full timeout
        # on unthrottled systems waiting for events that never arrive.
        try:
            out = subprocess.check_output(
                ["pmset", "-g", "therm"],
                text=True,
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        sched_limits = [int(value) for value in re.findall(r"(?:Scheduler|Speed)_Limit[^0-9]*(\d+)", out, flags=re.IGNORECASE)]
        warning_levels = [int(value) for value in re.findall(r"warning level[^0-9]*(\d+)", out, flags=re.IGNORECASE)]
        if sched_limits:
            limit = min(sched_limits)
            warning = max(warning_levels) if warning_levels else 0
            return max(self._thermal_estimate, min(95.0, 55.0 + (100 - limit) * 0.55 + warning * 4.0))
        if warning_levels:
            warning = max(warning_levels)
            return max(self._thermal_estimate, min(95.0, 62.0 + warning * 6.0))
        return None

    def _estimate_thermal(
        self,
        time_in_bound: float,
        x_used: int,
        k_used: int,
        ram_headroom_mb: float,
        ssd_read_rate_mb: float,
    ) -> float:
        x_scale = min(1.0, x_used / max(float(configs.X_MAX), 1.0))
        k_scale = min(1.0, k_used / max(float(configs.K_MAX), 1.0))
        time_scale = min(1.0, time_in_bound / 4.0)
        ram_capacity = max(float(configs.EXPERT_RAM_MB * configs.X_MAX), 1.0)
        ram_scale = 1.0 - min(1.0, ram_headroom_mb / ram_capacity)
        io_scale = min(1.0, ssd_read_rate_mb / 2048.0)
        target = 49.0 + 34.0 * (0.3 * x_scale + 0.2 * k_scale + 0.28 * time_scale + 0.17 * ram_scale + 0.05 * io_scale)
        previous = self.history[-1].thermal_state if self.history else self._thermal_estimate
        alpha = 0.3 if target >= previous else 0.12
        estimate = previous + (target - previous) * alpha
        self._thermal_estimate = max(42.0, min(92.0, estimate))
        return self._thermal_estimate

    def _read_ram(self) -> float:
        try:
            out = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
            page_size = 16384
            free = inactive = 0
            for line in out.splitlines():
                if line.startswith("Pages free:"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages inactive:"):
                    inactive = int(line.split(":")[1].strip().rstrip("."))
            return (free + inactive) * page_size / (1024 * 1024)
        except Exception:
            return float(configs.EXPERT_RAM_MB * 3)

    def _read_ssd_rate(self) -> float:
        try:
            out = subprocess.check_output(["iostat", "-d", "-K", "disk0"], text=True, stderr=subprocess.DEVNULL)
            lines = [
                line
                for line in out.splitlines()
                if line.strip() and not line.lstrip().startswith("disk")
            ]
            if lines:
                parts = lines[-1].split()
                if len(parts) >= 3:
                    return float(parts[2])
        except Exception:
            pass
        return 0.0

