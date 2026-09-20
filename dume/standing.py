"""Expert standing, keyed by the ROUTING unit: (expert, cluster).

Standing is a PURE RATE — no confidence, no discount:

    S_ik = sum_delta / sum_tokens        nats saved per emitted token

Confidence lives in exactly one place in this system, the Timeline A/B gate, and
it is Central's reliability that supplies it. It has no business inside an
expert's score: a z-discount folded into S made every consumer — ranking,
migration, note ordering at deployment — read a number that was part
measurement and part uncertainty penalty, and at n == 1 the population variance
was identically 0, so the discount vanished exactly where it was needed most.

sum_tokens is the count of tokens the expert EMITTED, never the input span the
router handed it — the span is the router's choice and must not move the
expert's rank.

    D[c]    = size[c] / sum(size)        domain rank, frozen at formation
    r[e]    = sum_c D[c] * S[e,c]        OVERALL rank, consumed by alloc.sample()

Domain rank cannot come from live traffic: tau breathes toward EQUAL presence
(balanced load is the fixed point), so presence rate is actively driven to
uniform and carries no domain information. Formation membership is the only
honest source.

Membership is MEASURED, and it is a RANK-AND-FILL, not a threshold (Aman,
2026-09-10). After the testing sweeps there is a rate for every expert, and the
sqrt bracket already says how the pool divides:

    GENERAL_EXPERTS  = ceil(sqrt(E))            = 10   the best, kept general
    CENTROID_EXPERTS = floor(sqrt(E - general)) =  9   seats per centroid

So: rank the pool by overall rate, keep the top GENERAL_EXPERTS as GENERAL —
they are good across the board, which is what general MEANS — and send everyone
else down into the centroid their own row points at hardest, best claimant
first, until that centroid's earned seats are full. Anyone who cannot be placed
falls back to GENERAL.

There is NO sign test and no 0.90 band on membership. Both were wrong: the band
came from the geometry, where it compares centroid DIRECTIONS, and standing is
a rate, not a direction. Clipping negatives before normalising meant that while
every measured rate sat below zero — which is exactly what 135 batches produced,
twelve pairs all at -0.002 to -0.005 nats/token — every expert returned the zero
vector, nobody could hold a home, migration seated nobody, the curriculum could
never leave TEST, and the whole geometry layer stayed inert forever. A relative
ranking has no such fixed point: somebody is always the best 10, and the rest
always point somewhere.

The general class is decided ONCE, when testing ends, and then LOCKED (Aman,
2026-09-10): "we don't compare generals, we keep generals and train, then no
more." A general is not re-ranked against the pool every migrate — it is kept
and it trains. Re-ranking them was churn dressed up as adaptation: simulated at
the measured noise (per-batch delta std 1.12), the top ten stabilises to 8.8/10
carry-over even when NO expert is actually better than any other, so a settling
general set is not evidence the set is right, and a moving one is not evidence
it is learning. Freezing removes the question.

Everyone else keeps migrating. A centroid member that performs elsewhere still
moves, and an unplaced expert (there are more experts than earned seats) is
GENERAL by residue, not by election — it is still ranked and still placeable.

Only experts with MIN_MOVE_OBS observations are ranked at all. An expert with no
evidence is GENERAL by default, which is also what makes it the dormant trial.

Class size per centroid is capped at CENTROID_EXPERTS (anti-dominance); over
the cap, the worst standings fall back to general rather than displacing into
a cluster they do not point at. Membership has exactly ONE writer: migrate().
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C

# p4: ranking is a "normalized weighted sum" of correctness, hallucination and
# processing time. Aman named the three terms; he did NOT name the coefficients.
#
# W_HALLUC IS NOT A COEFFICIENT ANY MORE — it is MEASURED (Aman, 2026-09-20:
# "w_hall is score we get from real isn't it?"). It was the last hand-set number
# in the ranking, and the file's own comment admitted it: "how much a
# hallucination should cost against an equal correctness is a judgement ...
# VALUE NEEDS AMAN". It isn't a judgement. Correctness and hallucination are the
# same per-token gradient read on either side of zero, both against y, which
# came off disk — so the pool's own books say what a hallucinated nat costs
# against a correct one. See Standing.w_halluc().
W_CORRECT, W_TIME = 1.0, 1.0

GENERAL = -1      # performance never concentrated: candidate everywhere, at home nowhere
MIN_MOVE_OBS = 2  # a rate needs more than one sample before it may move a seat


class Standing:
    def __init__(self, n_clusters: int, n_experts: int = C.E):
        self.C, self.E = int(n_clusters), int(n_experts)
        self.n = np.zeros((self.E, self.C), dtype=np.float64)
        self.sd = np.zeros((self.E, self.C), dtype=np.float64)
        self.sd2 = np.zeros((self.E, self.C), dtype=np.float64)
        self.sn = np.zeros((self.E, self.C), dtype=np.float64)
        # p4: ranking is a normalised weighted sum of CORRECTNESS, HALLUCINATION
        # and PROCESSING TIME. The first two are the same per-token gradient read
        # on either side of zero (reward._split); the third is measured wall time.
        self.sc = np.zeros((self.E, self.C), dtype=np.float64)   # weighted d>0
        self.sh = np.zeros((self.E, self.C), dtype=np.float64)   # weighted d<0, magnitude
        self.ssec = np.zeros(self.E, dtype=np.float64)           # seconds spent
        self.stok = np.zeros(self.E, dtype=np.float64)           # tokens emitted in them
        # cold: nobody has performed anywhere, so nobody points at a centroid.
        # Every expert is a candidate everywhere until it earns a home.
        self.assigned = np.full(self.E, GENERAL, dtype=np.int64)
        self.elite: set = set()      # the locked general class; empty until testing ends
        self.moves = 0

    # ── reading ─────────────────────────────────────────────────────────────
    def score(self, eid: int, cid: int) -> Optional[float]:
        """Nats saved per emitted token, or None when this pair has never run."""
        if self.n[eid, cid] <= 0 or self.sn[eid, cid] <= 0:
            return None
        return float(self.sd[eid, cid] / self.sn[eid, cid])

    @staticmethod
    def _unit(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Min-max onto [0,1] over the MEASURED experts only, so three quantities
        in different units (nats, nats, tokens/second) can be summed at all. A
        flat component contributes 0.5 to everyone, i.e. nothing."""
        out = np.full(v.shape, 0.5)
        if not mask.any():
            return out
        lo, hi = float(v[mask].min()), float(v[mask].max())
        out[mask] = 0.5 if hi - lo < 1e-12 else (v[mask] - lo) / (hi - lo)
        return out

    def w_halluc(self) -> float:
        """What a hallucinated nat costs against a correct one, read off the
        pool's own books: total nats hallucinated / total nats saved.

        Both sums are accumulated by observe() from d = b - a against real y, so
        this is a measurement, not a preference. On the 1566-batch run the pool
        hallucinated 331.0 nats for every 105.1 it saved, so the weight lands
        near 3.15 and the ranking leans on NOT hallucinating in exactly the
        proportion the data says it should.

        No clamp: wsum normalises, so a large weight simply makes the ranking
        non-hallucination-dominated, which is the correct response to a pool
        that mostly hurts. Falls back to 1.0 (the neutral pair) only when one
        side has no evidence at all — a cold pool has nothing to measure."""
        c, h = float(self.sc.sum()), float(self.sh.sum())
        return (h / c) if c > 0.0 and h > 0.0 else 1.0

    def _overall(self, M: np.ndarray, d: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Domain-weighted overall rate, with an unmeasured cluster imputed from
        the POOL rather than from zero.

        This was `M @ d`, a dot product over all C clusters in which a cell the
        expert had never been tried in contributed exactly 0.0. 202 of the 222
        measured cells in the live state are negative, so every gap was a free
        win: measured on that state, pearson(overall, clusters measured) =
        -0.283, p=0.0044. The metric paid for ignorance — and it is the metric
        that elects the general class, whose ten members had 4-9 observations
        each.

        Renormalising over the measured clusters instead would fix that and
        break something load-bearing: it makes "good in one small domain"
        identical to "good in one big domain", and weighting by domain size is
        exactly what `general` MEANS here. So the gap is filled with what the
        POOL does in that cluster — the best available estimate of an unknown
        cell — which is neutral by construction, while every cluster keeps its
        domain weight."""
        mask = np.asarray(mask, dtype=bool)
        col_n = mask.sum(0)
        col_mean = np.divide((M * mask).sum(0), col_n, where=col_n > 0,
                             out=np.zeros(self.C, dtype=np.float64))
        filled = np.where(mask, M, col_mean[None, :])
        return filled @ d

    def rank(self, domain: np.ndarray) -> Dict[int, float]:
        """r[e] = sum_c D[c] * S[e,c] over the clusters where e has run. One
        number per expert, comparable across the whole pool: good in a big
        domain beats equally good in a small one, and an expert that is
        moderate across several large domains outranks a specialist in one
        small one — which is what `general` means, so the ranking and the
        general centroid agree instead of being two mechanisms."""
        d = self._domain(domain)
        seen = self.n > 0
        div = np.maximum(self.sn, 1e-9)
        cor = self._overall(np.divide(self.sc, div, where=seen, out=np.zeros_like(self.sc)), d, seen)
        hal = self._overall(np.divide(self.sh, div, where=seen, out=np.zeros_like(self.sh)), d, seen)
        tps = self.throughput()
        m = seen.any(axis=1)
        # THREE SEPARATE TERMS (Aman, 2026-09-20: "ranking rewards non
        # hallucination, most processing in less time and efficiency").
        #
        #   efficiency       unit(cor)      nats the expert SAVED per token
        #   non-hallucination 1 - unit(hal) nats it COST, rewarded for being low
        #   processing/time  unit(tps)      tokens per second
        #
        # Correctness and hallucination used to share one normaliser and enter
        # as (cor - hal), so at W_HALLUC = 1.0 the two terms cancelled
        # algebraically and the "reward non-hallucination" half of the equation
        # did nothing. Each term now carries its own min-max, so every weight is
        # live at 1.0 and W_HALLUC > 1 makes volatility cost more rather than
        # being the only way it costs anything at all.
        w_h = self.w_halluc()
        wsum = W_CORRECT + w_h + W_TIME
        score = (W_CORRECT * self._unit(cor, m)
                 + w_h * (1.0 - self._unit(hal, m))
                 + W_TIME * self._unit(tps, m)) / wsum
        return {int(e): float(score[e]) for e in range(self.E) if m[e]}

    def members(self, cid: int) -> List[int]:
        return [int(e) for e in np.where(self.assigned == cid)[0]]

    def generals(self) -> List[int]:
        return [int(e) for e in np.where(self.assigned == GENERAL)[0]]

    def ranked(self, cid: int) -> List[Tuple[int, float]]:
        """Members + generals with evidence in this cluster, best first: (eid, S)."""
        out = [(e, self.score(e, cid)) for e in self.members(cid) + self.generals()]
        return sorted([(e, s) for e, s in out if s is not None], key=lambda x: -x[1])

    def trial(self, cid: int, exclude: set) -> Optional[int]:
        """The least-measured eligible expert — the dormant slot. Generals are
        the dormant pool now: an expert with no measured territory is general by
        definition, and this seat is the only way it ever gets measured."""
        pool = [e for e in self.members(cid) + self.generals() if e not in exclude]
        if not pool:
            pool = [e for e in range(self.E) if e not in exclude]
        if not pool:
            return None
        return int(min(pool, key=lambda e: (self.n[e, cid], e)))

    def total_n(self) -> float:
        return float(self.n.sum())

    # ── the two writers ─────────────────────────────────────────────────────
    def observe(self, eid: int, cid: int, delta: float, n_emitted: int,
                correct: float = 0.0, halluc: float = 0.0, seconds: float = 0.0) -> None:
        self.n[eid, cid] += 1.0
        self.sd[eid, cid] += float(delta)
        self.sd2[eid, cid] += float(delta) ** 2
        self.sn[eid, cid] += float(max(int(n_emitted), C.EXPERT_GEN_TOKENS))
        self.sc[eid, cid] += float(correct)
        self.sh[eid, cid] += float(halluc)
        if seconds and seconds > 0:
            self.ssec[eid] += float(seconds)
            self.stok[eid] += float(max(int(n_emitted), C.EXPERT_GEN_TOKENS))

    def throughput(self) -> np.ndarray:
        """Tokens per second, per expert, measured. Zero where never timed."""
        return np.divide(self.stok, np.maximum(self.ssec, 1e-9),
                         out=np.zeros(self.E), where=self.ssec > 0)

    def rates(self) -> np.ndarray:
        """The raw rate matrix, zeroed where a pair lacks MIN_MOVE_OBS. Not
        clipped: a negative rate is a real measurement and it still ORDERS."""
        r = np.divide(self.sd, np.maximum(self.sn, 1e-9))
        return np.where(self.n >= MIN_MOVE_OBS, r, 0.0)

    def rankable(self, eid: int) -> bool:
        """Has this expert a cell with enough evidence to be ranked at all?"""
        return bool((self.n[eid] >= MIN_MOVE_OBS).any())

    def settle_generals(self, domain=None) -> List[int]:
        """Elect the general class for the first time, when the curriculum
        leaves TEST — the one moment the whole pool has been swept equally and
        is comparable on equal evidence.

        It is NOT a lock (Aman, 2026-09-20: "generals aren't unreachable elites,
        they are just centroids with an unchangeable space; if someone performs
        better it can replace it"). The SPACE is fixed at GENERAL_EXPERTS; the
        membership is re-decided by migrate() on every pass, so an expert that
        out-performs an incumbent takes its seat and the incumbent falls back to
        a centroid. Locking it froze ten experts that had 4-9 observations each
        and then starved them of every further seat."""
        if self.elite:
            return sorted(self.elite)
        d = self._domain(domain)
        order = sorted((e for e in range(self.E) if self.rankable(e)),
                       key=lambda e: -float(self._overall(self.rates(), d, self.n >= MIN_MOVE_OBS)[e]))
        self.elite = set(order[: C.GENERAL_EXPERTS])
        return sorted(self.elite)

    def _domain(self, domain) -> np.ndarray:
        return (np.full(self.C, 1.0 / max(self.C, 1)) if domain is None
                else np.asarray(domain, dtype=np.float64).reshape(-1)[: self.C])

    def migrate(self, chains=None, cap=None, domain=None) -> List[Tuple[int, int, int]]:
        """Rank and fill. The only writer of membership.

        Rank every measured expert by overall rate (domain-weighted over the
        clusters it has actually been measured in). The top GENERAL_EXPERTS are
        the general class. The rest, best first, take a seat in whichever
        centroid their own row favours that still has room; no room anywhere ->
        GENERAL by residue.

        The general class is re-decided HERE, every pass. It is a space of fixed
        size, not a set of fixed members: a newcomer that out-ranks an incumbent
        takes its place and the incumbent drops to a centroid seat. The previous
        version excluded a settled elite from `order` entirely, so it could
        neither be displaced nor re-seated, and since _centroid_batch draws only
        from members() those ten experts then took zero of 3550 observations in
        1566 batches.

        The per-centroid cap is anti-dominance AND growth: a centroid holds only
        the seats it has EARNED (geometry.capacity(), floor(sqrt(input seen)),
        ceilinged at CENTROID_EXPERTS). cap=None means the ceiling everywhere."""
        caps = ([int(C.CENTROID_EXPERTS)] * self.C if cap is None
                else [int(x) for x in np.asarray(cap).reshape(-1)[: self.C]])
        d = self._domain(domain)
        rate = self.rates()
        overall = self._overall(rate, d, self.n >= MIN_MOVE_OBS)
        order = sorted((e for e in range(self.E) if self.rankable(e)),
                       key=lambda e: -float(overall[e]))
        # the top of the order IS the general class, re-decided here every pass.
        # They stand aside from the fill, so the centroid seats go to the rest.
        if order:
            self.elite = set(order[: C.GENERAL_EXPERTS])
            order = order[C.GENERAL_EXPERTS:]
        want: Dict[int, int] = {e: GENERAL for e in range(self.E)}
        free = list(caps)
        for e in order:
            for c in sorted((c for c in range(self.C) if self.n[e, c] >= MIN_MOVE_OBS),
                            key=lambda c: -rate[e, c]):
                if free[c] > 0:
                    want[e], free[c] = c, free[c] - 1
                    break
        moves: List[Tuple[int, int, int]] = []
        for e, to in want.items():
            frm = int(self.assigned[e])
            if to != frm:
                self._move(e, frm, to, moves, chains)
        return moves

    def _move(self, e: int, frm: int, to: int, moves, chains) -> None:
        self.assigned[e] = to
        self.moves += 1
        moves.append((e, frm, to))
        if chains is not None:
            chains.record_move(e, frm, to)

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, object]:
        return {"n": self.n, "sd": self.sd, "sd2": self.sd2, "sn": self.sn,
                "sc": self.sc, "sh": self.sh, "ssec": self.ssec, "stok": self.stok,
                "assigned": self.assigned, "moves": self.moves, "C": self.C, "E": self.E,
                "elite": sorted(self.elite)}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Standing":
        s = Standing(int(d["C"]), int(d["E"]))
        for k in ("n", "sd", "sd2", "sn", "assigned"):
            setattr(s, k, np.asarray(d[k]).copy())
        for k in ("sc", "sh", "ssec", "stok"):          # additive: older states lack them
            if k in d:
                setattr(s, k, np.asarray(d[k]).copy())
        s.moves = int(d.get("moves", 0))
        s.elite = {int(x) for x in d.get("elite", [])}
        return s
