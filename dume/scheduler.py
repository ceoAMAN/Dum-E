"""THE owner of concurrency and residency. Nothing else clamps, defaults or
re-derives a bound (rule 6).

    usable   = R - sqrt(R) - central - gate            Aman's law; sqrt(R) in GB
    k_fit    = floor( usable / expert_peak )
    k_tier   = floor( sqrt(R_GB) )                     Aman's RAM tiers: 16->4, 64->8
    k_max    = min(k_fit, k_tier, MAX_CLUSTERS)
    span_max = ( usable - k_max*expert_peak ) / slope  the SAME law, for the transient

span_max is the third bound and it comes off the same `usable`. Once k_max experts
are resident, whatever `usable` still has is what a gradient step may spend, and a
gradient step costs a MEASURED ~12 MB per prompt token (models.measure_update_slope_mb)
because it stores activations for every layer over the whole sequence. Unbounded, a
1,024-token prompt asks for 13 GB and the OS SIGKILLs the trainer. This is a physical
clamp on apex-nadir's t_hi exactly as k_max is a physical clamp on its k.

sqrt(R) is the headroom reserve — 4 GB of the 16 on this M4 — and it is the same
sqrt bracket k_tier uses. k_tier binds in practice here (k_max 4), so the pool
commits 6-8 GB against Metal's 11.8 GB ceiling; the ceiling is printed alongside
and a warning fires if a commitment ever crosses it.

expert_peak is MEASURED at boot. If k_fit < 1 the boot halts.

k = min(k_wanted, k_max). When k_wanted > k_max the scheduler clamps k and the
router keeps the MOST IMPORTANT experts (Aman: "when k is minimum gating
prioritises important experts most"). Residency is LRU over k_max slots;
experts stay resident after work until displaced.
"""
from __future__ import annotations

import math
from typing import Iterable, List

from . import config as C
from .models import ExpertPool, active_mb, thermal_state, total_ram_mb, working_set_mb


class Scheduler:
    def __init__(self, pool: ExpertPool, expert_peak_mb, central_mb: float, gate_mb: float,
                 update_slope_mb: float = 0.0, slot_mb: float = 0.0):
        self.pool = pool
        self.thermal = None      # a ThermalRegulator, set by System._wire()
        R = total_ram_mb()
        ws = working_set_mb()
        # Aman's law:  space for k  =  R - sqrt(R) - central - gate
        # sqrt(R) is in GB (16 GB -> a 4 GB reserve), the same bracket k_tier uses.
        reserve = math.sqrt(R / 1024.0) * 1024.0
        usable = R - reserve - central_mb - gate_mb
        self.k_tier = max(1, int(math.floor(math.sqrt(R / 1024.0))))
        if expert_peak_mb is None:                      # no experts this run: nothing to fit
            self.k_fit, self.k_max, self.span_max = 0, 0, C.EXPERT_GEN_TOKENS
            print(f"[scheduler] RAM {R:.0f} MB | reserve sqrt {reserve:.0f} | central {central_mb:.0f} | "
                  f"gate {gate_mb:.0f} | usable {usable:.0f} | (Metal ws {ws:.0f}) | "
                  f"experts not measured (none needed)")
            return
        # The base is SHARED: it is paid once, and a seat costs only its adapter
        # plus Adam's moments (~105 MB), not another 1.2 GB copy of the same
        # frozen weights. That redundancy was the whole of k_fit.
        self.slot_mb = float(slot_mb) if slot_mb > 0 else float(expert_peak_mb)
        self.k_fit = int(max(0.0, usable - expert_peak_mb) // max(1.0, self.slot_mb))
        self.k_max = max(0, min(self.k_fit, self.k_tier, C.MAX_CLUSTERS))
        # the same `usable`, for the transient: what is left once k_max are resident,
        # divided by the measured cost of one prompt token in the backward pass.
        # The sequence that gets backpropped is span + orientation header + the
        # expert's own generated text, so two TARGET_MAX_TOKENS come off the top
        # before what remains is the span.
        free = usable - expert_peak_mb - self.k_max * self.slot_mb
        self.span_max = (int(max(C.EXPERT_GEN_TOKENS, free / update_slope_mb - 2 * C.TARGET_MAX_TOKENS))
                         if update_slope_mb > 0 else 1 << 30)
        committed = central_mb + gate_mb + expert_peak_mb + self.k_max * self.slot_mb
        print(f"[scheduler] RAM {R:.0f} MB | reserve sqrt {reserve:.0f} | central {central_mb:.0f} | "
              f"gate {gate_mb:.0f} | usable {usable:.0f} | base {expert_peak_mb:.0f} + "
              f"{self.slot_mb:.0f}/seat -> "
              f"k_fit {self.k_fit}, k_tier {self.k_tier} => k_max {self.k_max} "
              f"| commits {committed:.0f} of Metal ws {ws:.0f} "
              f"| free {free:.0f} / {update_slope_mb:.1f} MB per tok => span_max {self.span_max}")
        if committed > ws:
            print(f"[scheduler] WARNING commits {committed:.0f} MB > Metal working set {ws:.0f} MB "
                  f"— the GPU refuses allocations past that ceiling")
        if self.k_max < 1:
            raise RuntimeError("scheduler: not one expert fits alongside central+gate — refusing to run")

    @property
    def k_thermal(self) -> float:
        """How many experts the DEVICE is willing to run right now.

        Each unit of thermal PRESSURE takes another root of the RAM bound —
        the same sqrt bracket k_tier, GENERAL_EXPERTS and capacity() are built
        from, so no new constant enters. On k_max=4 that is 4.00 nominal, 2.00
        fair, 1.59 serious, 1.41 critical: at nominal the device asks for
        nothing and the bound is exactly the physical one, and it tightens
        geometrically as the OS reports heat.

        The levels are ordinal — macOS publishes no degrees — so a geometric
        backoff is the only honest shape; a linear map would need a scale
        nobody measured.

        Pressure is the raw level measured against where this machine NORMALLY
        sits, scaled by how far its last move ran above its own mean rate —
        see ThermalRegulator. At the baseline the pressure is 0 and the bound
        is exactly the physical one, so a machine that simply runs warm is
        never throttled for it, and neither is one that warms at the pace it
        always warms at.

        The regulator RAMPS k toward that bound at the rate the machine itself
        moves, instead of snapping to it, so a single warm read no longer costs
        a residency reshuffle."""
        lvl = thermal_state()
        if self.thermal is not None:
            self.thermal.observe(lvl)
            return self.thermal.k_thermal(float(self.k_max), lvl)
        return float(self.k_max) ** (1.0 / (1.0 + lvl))

    def clamp(self, k_wanted: int) -> int:
        return max(1, min(int(k_wanted), self.k_max))

    def ensure(self, eids: Iterable[int]) -> List[int]:
        """Make these experts resident, evicting LRU non-needed ones to stay
        within k_max. Returns the resident ids in request order."""
        want = list(dict.fromkeys(int(e) for e in eids))[: self.k_max]
        for e in want:
            while e not in self.pool.resident and len(self.pool.resident) >= self.k_max:
                victims = [x for x in self.pool.resident if x not in want]
                if not victims:
                    break
                self.pool.unload(min(victims, key=lambda x: self.pool.last_used.get(x, 0.0)))
            self.pool.load(e)
        return [e for e in want if e in self.pool.resident]

    def status(self) -> str:
        return f"resident={sorted(self.pool.resident)} active_mb={active_mb():.0f}"
