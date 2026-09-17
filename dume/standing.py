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
# THE ONE THAT MATTERS IS W_HALLUC. Correctness and hallucination are the same
# per-token gradient read on either side of zero, in the SAME units (nats), so at
# W_CORRECT == W_HALLUC the pair collapses ALGEBRAICALLY to correct - halluc,
# which is exactly the net delta the system already had — an expert that gains 3
# nats and loses 3 ranks identically to one that does nothing. Only W_HALLUC > 1
# makes volatility cost anything, and how much a hallucination should cost
# against an equal correctness is a judgement about the product, not a fact the
# data can supply. VALUE NEEDS AMAN; 1.0 is the neutral default, not a choice.
W_CORRECT, W_HALLUC, W_TIME = 1.0, 1.0, 1.0

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
        cor = np.divide(self.sc, div, where=seen, out=np.zeros_like(self.sc)) @ d
        hal = np.divide(self.sh, div, where=seen, out=np.zeros_like(self.sh)) @ d
        tps = self.throughput()
        m = seen.any(axis=1)
        # normalised weighted sum (p4). Hallucination enters NEGATED — it is the
        # only one of the three where more is worse. Equal weights: Aman named the
        # three terms, not their coefficients, and inventing a split would be a
        # tuned constant closing an adaptive loop.
        # correctness and hallucination share ONE normaliser: they are the same
        # quantity in the same units, and scaling them apart would invent an
        # asymmetry the data never showed. Throughput is unrelated, so it gets
        # its own. Weights are applied AFTER, where W_HALLUC can do its work.
        scale = max(float(np.abs(cor[m]).max()) if m.any() else 0.0,
                    float(np.abs(hal[m]).max()) if m.any() else 0.0, 1e-12)
        grad = (W_CORRECT * cor - W_HALLUC * hal) / scale
        wsum = W_CORRECT + W_HALLUC + W_TIME
        score = ((W_CORRECT + W_HALLUC) * self._unit(grad, m)
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
        """Elect the general class and LOCK it. Called once, when the curriculum
        leaves TEST — every expert has been swept the same number of times, so
        this is the one moment the whole pool is comparable on equal evidence.
        After this they are kept and trained, never compared again."""
        if self.elite:
            return sorted(self.elite)
        d = self._domain(domain)
        rate = self.rates()
        order = sorted((e for e in range(self.E) if self.rankable(e)),
                       key=lambda e: -float(rate[e] @ d))
        self.elite = set(order[: C.GENERAL_EXPERTS])
        return sorted(self.elite)

    def _domain(self, domain) -> np.ndarray:
        return (np.full(self.C, 1.0 / max(self.C, 1)) if domain is None
                else np.asarray(domain, dtype=np.float64).reshape(-1)[: self.C])

    def migrate(self, chains=None, cap=None, domain=None) -> List[Tuple[int, int, int]]:
        """Rank and fill. The only writer of membership.

        Rank every measured expert by overall rate (domain-weighted, the same
        number alloc ranks by). The top GENERAL_EXPERTS stay GENERAL. The rest,
        best first, take a seat in whichever centroid their own row favours that
        still has room; no room anywhere -> GENERAL.

        Once settle_generals() has run, the elite is excluded outright: it is
        neither ranked nor placed, and it cannot lose its class to a newcomer.

        The per-centroid cap is anti-dominance AND growth: a centroid holds only
        the seats it has EARNED (geometry.capacity(), floor(sqrt(input seen)),
        ceilinged at CENTROID_EXPERTS). cap=None means the ceiling everywhere."""
        caps = ([int(C.CENTROID_EXPERTS)] * self.C if cap is None
                else [int(x) for x in np.asarray(cap).reshape(-1)[: self.C]])
        d = self._domain(domain)
        rate = self.rates()
        measured = [e for e in range(self.E)
                    if self.rankable(e) and e not in self.elite]
        order = sorted(measured, key=lambda e: -float(rate[e] @ d))
        # before the class is locked the skim is PROVISIONAL: the current best
        # stand aside so the seats go to the rest. After settle_generals() the
        # elite is out of `order` entirely and nothing skims.
        if not self.elite:
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
