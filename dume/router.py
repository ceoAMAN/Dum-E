"""Routing: composition -> present clusters -> experts -> CONTIGUOUS spans.

Gating owns token allocation (rule 10). It emits (expert, start, end) and the
run loop only cuts at those offsets. Spans are contiguous (rule 17): grouping
by cluster happens in the ORDER of spans, never by gathering scattered indices.

Selection within a cluster is by STANDING (grounded), with the route head's
learned preference as a tiebreak. One slot per batch, when k >= 2, goes to the
least-measured expert of the home cluster at SPAN_MIN tokens — the dormant
trial. It is scored and updated like everyone else; nothing is retired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from .geometry import Geometry
from .models import Gate
from .scheduler import Scheduler
from .standing import Standing


@dataclass
class Selection:
    eid: int
    cid: int
    tier: str
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
    k_wanted: int
    k: int
    selections: List[Selection] = field(default_factory=list)


class Router:
    def __init__(self, gate: Gate, geometry: Geometry, standing: Standing, scheduler: Scheduler):
        self.gate, self.geo, self.standing, self.sched = gate, geometry, standing, scheduler

    def plan(self, text: str) -> Plan:
        ids = self.gate.encode(text)
        H = self.gate.hidden(ids)
        T = len(ids)
        pooled = H.mean(0)
        w, assign, _, mean_sims = self.geo.compose(H)
        home = self.geo.home(w)
        present = self.geo.present(w, mean_sims)
        # one seat per present cluster PLUS the trial seat (Aman: "one slot always
        # reserved for an unmeasured expert"), bounded by the sqrt(C) band
        k_wanted = max(1, min(len(present) + 1, C.k_upper(self.geo.C)))
        k = self.sched.clamp(k_wanted)
        route = self.gate.route_logits(pooled)
        chosen: set = set()
        picks: List[tuple] = []           # (eid, cid, tier, trial)
        seats = k - 1 if k >= 2 else k          # the last seat is the trial's
        for cid in present[:seats]:
            eid = self._pick(cid, route, chosen)
            if eid is None:
                continue
            chosen.add(eid)
            picks.append((eid, cid, self.geo.tier(home, cid), False))
        if k >= 2:
            routed = present[0]                     # the trial is measured where the input actually routed
            t = self.standing.trial(routed, chosen)
            if t is not None:
                picks.append((t, routed, self.geo.tier(home, routed), True))
        plan = Plan(ids=ids, H=H, pooled=pooled, w=w, assign=assign, home=home, present=present,
                    k_wanted=k_wanted, k=k)
        plan.selections = self._spans(picks, w, assign, T)
        return plan

    def _pick(self, cid: int, route: np.ndarray, chosen: set) -> Optional[int]:
        """known-good > unmeasured > known-bad. An expert that has HURT in this
        cluster must not keep its seat just because nobody else has been
        measured yet; the unmeasured get their turn first."""
        ranked = [(e, s) for e, s in self.standing.ranked(cid) if e not in chosen]
        good = [(e, s) for e, s in ranked if s > 0]
        if good:
            top = good[0][1]
            near = [e for e, s in good if s >= top - 1e-9]
            return int(max(near, key=lambda e: route[e]))       # tiebreak by learned preference
        measured = {e for e, _ in ranked}
        pool = [e for e in self.standing.members(cid) + self.standing.generals()
                if e not in chosen and e not in measured]
        if pool:
            return int(max(pool, key=lambda e: route[e]))
        if ranked:
            return int(ranked[0][0])                             # least bad
        rest = [e for e in range(self.standing.E) if e not in chosen]
        return int(max(rest, key=lambda e: route[e])) if rest else None

    def _spans(self, picks: List[tuple], w: np.ndarray, assign: np.ndarray, T: int) -> List[Selection]:
        """Contiguous consecutive spans, ordered by where each cluster's tokens
        sit, sized by composition x tier weight, min SPAN_MIN each."""
        if not picks or T == 0:
            return []
        com = []
        for eid, cid, tier, trial in picks:
            pos = np.where(assign == cid)[0]
            com.append(float(pos.mean()) if len(pos) else T / 2.0)
        order = np.argsort(com)
        raw = []
        for i in order:
            eid, cid, tier, trial = picks[i]
            weight = float(w[cid]) * C.TIER_WEIGHT[tier]
            raw.append((eid, cid, tier, trial, weight))
        n = len(raw)
        min_each = min(C.SPAN_MIN, max(1, T // n))
        total_w = sum(max(r[4], 1e-6) for r in raw)
        spare = max(0, T - min_each * n)
        lengths = [min_each + int(spare * max(r[4], 1e-6) / total_w) for r in raw]
        lengths[-1] += T - sum(lengths)
        out, start = [], 0
        for (eid, cid, tier, trial, _), L in zip(raw, lengths):
            L = max(1, L)
            end = min(T, start + L)
            if trial:
                end = min(T, start + max(1, min(L, C.SPAN_MIN)))
            out.append(Selection(eid=eid, cid=cid, tier=tier, start=start, end=end, trial=trial))
            start = end
        if out and out[-1].end < T:
            out[-1].end = T
        return out

    def gate_step(self, plan: Plan, deltas: Dict[int, float]) -> float:
        return self.gate.train_step(plan.pooled, deltas)
