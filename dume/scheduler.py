"""THE owner of concurrency and residency. Nothing else clamps, defaults or
re-derives a bound (rule 6).

    k_fit  = floor( (R - sqrt(R) - central - gate - working) / expert_peak )
    k_tier = floor( sqrt(R_GB) )                       Aman's RAM tiers: 16->4, 64->8
    k_max  = min(k_fit, k_tier, MAX_CLUSTERS)

Every term is MEASURED at boot (expert peak, Central's CE-forward peak as the
working reserve) — "measure it, don't set it" — and the result is printed with
its inputs so a wrong number is visible, not silent. If k_fit < 1 the boot halts.

k = min(k_wanted, k_max). When k_wanted > k_max the scheduler clamps k and the
router keeps the MOST IMPORTANT experts (Aman: "when k is minimum gating
prioritises important experts most"). Residency is LRU over k_max slots;
experts stay resident after work until displaced.
"""
from __future__ import annotations

import math
from typing import Iterable, List

from . import config as C
from .models import ExpertPool, active_mb, total_ram_mb


class Scheduler:
    def __init__(self, pool: ExpertPool, expert_peak_mb, central_mb: float,
                 gate_mb: float, working_mb: float):
        self.pool = pool
        R = total_ram_mb()
        usable = R - math.sqrt(R) - central_mb - gate_mb - working_mb
        self.k_tier = max(1, int(math.floor(math.sqrt(R / 1024.0))))
        if expert_peak_mb is None:                      # no experts this run: nothing to fit
            self.k_fit, self.k_max = 0, 0
            print(f"[scheduler] RAM {R:.0f} MB | central {central_mb:.0f} | gate {gate_mb:.0f} | "
                  f"working {working_mb:.0f} | experts not measured (none needed)")
            return
        self.k_fit = int(usable // max(1.0, expert_peak_mb))
        self.k_max = max(0, min(self.k_fit, self.k_tier, C.MAX_CLUSTERS))
        print(f"[scheduler] RAM {R:.0f} MB | central {central_mb:.0f} | gate {gate_mb:.0f} | "
              f"working {working_mb:.0f} | expert peak {expert_peak_mb:.0f} -> k_fit {self.k_fit}, "
              f"k_tier {self.k_tier} => k_max {self.k_max}")
        if self.k_max < 1:
            raise RuntimeError("scheduler: not one expert fits alongside central+gate — refusing to run")

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
