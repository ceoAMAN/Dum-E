"""Routing: composition -> present clusters -> experts -> CONTIGUOUS spans.

Gating owns token allocation (rule 10). It emits (expert, start, end) and the
run loop only cuts at those offsets. Spans are contiguous (rule 17): grouping
by cluster happens in the ORDER of spans, never by gathering scattered indices.

Selection within a cluster is by STANDING (grounded). The route head decides
among candidates whose standing intervals OVERLAP — statistically tied, so the
grounded number has nothing left to say — and among the unmeasured. That is its
consumer; without one it would be a mechanism that learns and changes nothing.

One seat per batch, when k >= 2, goes to the least-measured expert of the routed
cluster — the dormant trial. Its span is apex-nadir's like everyone else's: a
dormant expert measured only at a fixed small span has a standing that is not
comparable to a seated one measured at its own allocation. It is scored and
updated like everyone else; nothing is retired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .alloc import AllocLaw, probe_sizes
from .geometry import Geometry
from .models import Gate
from .scheduler import Scheduler
from .standing import Standing


@dataclass
class Selection:
    eid: int
    cid: int
    start: int
    end: int
    trial: bool = False

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


@dataclass
class Plan:
    ids: List[int]
    H: np.ndarray
    pooled: np.ndarray
    w: np.ndarray
    assign: np.ndarray
    home: int
    present: List[int]
    inside: List[int]
    k_wanted: int
    k: int
    probe: bool = False
    selections: List[Selection] = field(default_factory=list)


class Router:
    def __init__(self, gate: Gate, geometry: Geometry, standing: Standing,
                 scheduler: Scheduler, alloc: AllocLaw):
        self.gate, self.geo, self.standing, self.sched = gate, geometry, standing, scheduler
        self.alloc = alloc

    def plan(self, text: str, probe: bool = False, curriculum=None) -> Plan:
        """`curriculum` is the TRAINING path: it is asked for the k experts once
        k and the routed cluster are known, and selection by standing is skipped
        entirely. The
        router still supplies everything else — composition, home, present, the
        spans, apex-nadir's allocation. Selection by standing (below) is the
        DEPLOYMENT path, where concentrating work on the best expert is the
        whole point; during training it is what starved 66 experts of any
        measurement at all."""
        ids = self.gate.encode(text)
        H = self.gate.hidden(ids)
        T = len(ids)
        pooled = H.mean(0)
        w, assign, _, mean_sims = self.geo.compose(H)
        home = self.geo.home(w)
        inside = self.geo.inside(mean_sims)
        # sqrt(C) admissibility band: at most band_hi centroids are live for one
        # input, however many its territory happens to contain. On C=10 that is 4.
        present = self.geo.present(w, mean_sims)[: self.geo.band()[1]]
        # k IS the OLS regression: k(T) = T/ALLOC(T) = T^(1-beta)/alpha. Nothing
        # else sets it. The scheduler's RAM bound still clamps afterwards because
        # that one is physical, not a policy.
        k_wanted = self.alloc.k(T)
        # k_effective blends the allocation law, the measured clock and the RAM
        # bound (Aman's general equation); sched.clamp stays because the RAM
        # bound is physical and a blend must never be allowed above it.
        k = self.sched.clamp(self.alloc.k_effective(T, self.sched.k_max))
        picks: List[tuple] = []           # (eid, cid, trial)
        if curriculum is not None:
            experts = curriculum.experts(k, present[0], self.standing,
                                         resident=getattr(self.sched.pool, 'resident', ()))
            # every seat is measured on the cluster the input actually routed to,
            # so the observations of a whole sweep are comparable across experts.
            routed = present[0]
            # The LAST seat of the batch is the trial, exactly as on the router
            # path. Marking every seat trial=False silently deleted the dormant
            # slot the moment training stopped routing: `_imitate` looks for
            # sel.trial and found none, so imitation ran 0 times in 1700 batches.
            # The curriculum still chooses WHICH experts sit; this only labels
            # one of them, so equal exposure is untouched.
            chosen = [int(e) for e in experts[:k]]
            picks = [(e, routed, i == len(chosen) - 1 and len(chosen) >= 2)
                     for i, e in enumerate(chosen)]
        else:
            route = self.gate.route_logits(pooled)
            chosen: set = set()
            seats = k - 1 if k >= 2 else k          # the last seat is the trial's
            for cid in present[:seats]:
                eid = self._pick(cid, route, chosen)
                if eid is None:
                    continue
                chosen.add(eid)
                picks.append((eid, cid, False))
            if k >= 2:
                routed = present[0]                 # the trial is measured where the input actually routed
                t = self.standing.trial(routed, chosen)
                if t is not None:
                    picks.append((t, routed, True))
        plan = Plan(ids=ids, H=H, pooled=pooled, w=w, assign=assign, home=home, present=present,
                    inside=inside, k_wanted=k_wanted, k=k, probe=probe)
        place = AllocLaw.placement(self.standing.rank(self.geo.domain()))
        # FRAGMENT CYCLES (Aman, 2026-09-20). k_effective is not only a count:
        # it is the number of experts handed to the gate "with instruction to
        # increase token fragment cycles on them, so then we avoid swapping k
        # experts on regular intervals". The law asked for k_wanted experts; RAM
        # allows k. The deficit is paid in CYCLES on the experts we are already
        # holding rather than in swaps, so the input still gets covered and the
        # adapter stays live between its own fragments.
        #
        # Deployment only. Training's lengths come from the probe schedule —
        # the probe IS the training allocation — and cycling there would spend
        # the run's wall clock on coverage it does not need.
        cycles = 1 if probe else max(1, k_wanted // max(k, 1))
        plan.selections = self._spans(picks, assign, T, probe, place, cycles)
        return plan

    def _pick(self, cid: int, route: np.ndarray, chosen: set) -> Optional[int]:
        """known-good > unmeasured > known-bad. An expert that has HURT in this
        cluster must not keep its seat just because nobody else has been
        measured yet; the unmeasured get their turn first. Among experts whose
        standing intervals overlap the leader's, the route head chooses — the
        grounded signal cannot separate them, so the learned one is consulted."""
        ranked = [(e, s) for e, s in self.standing.ranked(cid) if e not in chosen]
        good = [(e, s) for e, s in ranked if s > 0]
        if good:
            return int(good[0][0])                               # best measured rate, full stop
        measured = {e for e, _ in ranked}
        pool = [e for e in self.standing.members(cid) + self.standing.generals()
                if e not in chosen and e not in measured]
        if pool:
            return int(max(pool, key=lambda e: route[e]))
        if ranked:
            return int(ranked[0][0])                             # least bad
        rest = [e for e in range(self.standing.E) if e not in chosen]
        return int(max(rest, key=lambda e: route[e])) if rest else None

    def _spans(self, picks: List[tuple], assign: np.ndarray, T: int, probe: bool,
               place: Dict[int, float], cycles: int = 1) -> List[Selection]:
        """Split, then PAD. Anchors are contiguous and ordered by where each
        cluster's tokens actually sit; the LENGTH is apex-nadir's, not the
        router's — each expert gets the span its OWN rank earns it on the
        fitted curves, padded around its anchor and clamped to the input. Span
        size is a token-allocation question and allocation is fitted, not
        weighted by the cluster's relation to home.

        In PROBE mode (training) the lengths come from the probe schedule
        {t_lo, t_mid, t_hi} instead, cycled across the seats: the probe IS the
        training allocation, so exploring span sizes costs no extra runs."""
        if not picks or T == 0:
            return []
        picks = picks[:T]                       # never more spans than tokens
        com = []
        for eid, cid, trial in picks:
            pos = np.where(assign == cid)[0]
            com.append(float(pos.mean()) if len(pos) else T / 2.0)
        raw = [picks[i] for i in np.argsort(com)]
        n = len(raw)
        # apex-nadir explores span sizes, but not past what a gradient step can
        # afford: the backward pass is linear in prompt length (~12 MB/token) and
        # t_hi = T would ask for 13 GB on a long row. Same physical clamp as k_max.
        T_eff = min(T, self.sched.span_max)
        sizes = probe_sizes(T_eff) if probe else None
        # the probe IS the training allocation: cycling there would spend the
        # run's wall clock covering input the probe schedule is deliberately
        # sampling instead. The guard lives here as well as in plan() so the
        # invariant is local to the function that would break it.
        cycles = 1 if probe else max(1, int(cycles))
        out = []
        for i, (eid, cid, trial) in enumerate(raw):
            # rotate the probe across BATCHES as well as seats: at k=1 (the §7
            # identity fallback, which is the live state until the law fits)
            # `i` is always 0, so every probe would land on t_lo and the
            # envelopes could never be fitted over a range at all.
            L = sizes[(self.alloc.n_seen + i) % 3] if sizes else self.alloc.alloc(T_eff, place.get(eid, 0.5))
            L = max(1, min(int(L), T_eff))
            # each expert owns a contiguous REGION and reads it in `cycles`
            # fragments. Its fragments are emitted consecutively, so the adapter
            # is made live once and stays live for all of them — the swap is per
            # EXPERT, not per fragment, which is the whole point of paying the
            # deficit in cycles.
            lo, hi = (T * i) // n, (T * (i + 1)) // n
            # cycles are bounded by COVERAGE, not by the deficit alone: once an
            # expert's own region is tiled by its span there is nothing left to
            # read, and a bigger deficit would only buy duplicate passes. So the
            # instruction is "cover your region", and the deficit is the ceiling
            # on how hard it may try.
            need = max(1, -(-(hi - lo) // L))                 # ceil(region / span)
            reps = max(1, min(cycles, need))
            step = max(1, (hi - lo) // reps) if reps > 1 else 0
            for j in range(reps):
                start = min(lo + j * step, max(0, T - 1))
                end = min(T, start + L)
                start = max(0, end - L)                       # pad back off the end
                sel = Selection(eid=eid, cid=cid, start=start, end=end, trial=trial)
                if sel.n_tokens > 0 and not any(
                        o.eid == eid and o.start == start and o.end == end for o in out):
                    out.append(sel)                           # a region shorter than
        return out                                            # `cycles` stops repeating itself

    def gate_step(self, plan: Plan, deltas: Dict[int, float]) -> float:
        return self.gate.train_step(plan.pooled, deltas)
