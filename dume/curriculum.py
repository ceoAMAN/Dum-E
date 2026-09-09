"""The TRAINING schedule. Training does not route (Aman, 2026-09-10):

    "in training we would take batches of random experts, consistent batch in k,
     then done, new batch till all are done ... test then specialize, when we
     divide experts in batches in the centroids then train centroid-specific
     batches in rounds till how much token is allotted. Router based is for
     deployment."

Why it has to be this way, measured: over batches 77-126 the router gave e8, e1,
e78 and e29 sixteen, fourteen, eight and six observations while SIXTY-SIX experts
got none at all. Standing is a mean, and a mean over zero samples separates
nothing — so the router, whose whole job is to concentrate work on whoever is
already good, is precisely the wrong instrument for MEASURING who is good. It is
the right instrument at deployment, where concentrating is the point.

Two phases:

  TEST        the pool is shuffled and cut into consecutive groups of k. A group
              is CONSISTENT while it runs; when the pool is exhausted that is one
              sweep and the pool reshuffles. Every expert gets the same exposure,
              so position() is built on equal evidence for everyone.

  SPECIALIZE  homes exist now. Each centroid keeps its own rota of its members,
              and an input is served by the rota of the centroid it routes to,
              so a centroid's experts see that centroid's material. Rounds run
              until the token budget is spent.

Either way the chosen group is ORDERED so experts already holding the live
adapter go first.

The phase boundary is a MEASUREMENT, not a timer. Testing is over when every
expert is RANKABLE — has a cell with MIN_MOVE_OBS observations in it — because
that is the moment the whole pool is comparable on equal evidence, and it is the
one moment the general class can be elected fairly. MIN_SWEEPS is only a floor.

A sweep gives an expert one observation, but not one in a chosen CLUSTER: the
cluster is whatever the input routed to, and the live routing share is
[0.12, 0.36, 0.33, 0.19], so cells fill at very different rates. Simulated on
that share, 2 sweeps leaves 72% of the pool unrankable and 6 sweeps leaves 0%.
Hence coverage, not a sweep count, decides.

At that boundary `standing.settle_generals()` elects the general class and LOCKS
it — generals are kept and trained from then on, never re-compared. Migration
then places everyone else, and if it seats nobody the curriculum stays in test:
there is nothing to specialize into.

What this does NOT touch: composition, spans, apex-nadir allocation, scoring,
standing. The curriculum answers exactly one question — which experts sit in the
k seats — and the rest of the system is unchanged.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from . import config as C

TEST, SPECIALIZE = "test", "specialize"
MIN_SWEEPS = 2          # a sweep gives each expert ONE observation; one is not a mean


class Curriculum:
    def __init__(self, n_experts: int = C.E, seed: int = 0):
        self.E = int(n_experts)
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.cursor = 0
        self.sweeps = 0
        self.rounds = 0
        self.phase = TEST
        self.rota: Dict[int, List[int]] = {}      # cid -> its members, cycled
        self.rota_at: Dict[int, int] = {}

    # ── test: sweep the whole pool in consistent groups of k ────────────────
    def _reshuffle(self) -> None:
        self.order = list(range(self.E))
        self.rng.shuffle(self.order)
        self.cursor = 0

    def _sweep_batch(self, k: int) -> List[int]:
        if self.cursor >= len(self.order):
            self._reshuffle()
            if self.order:
                self.sweeps += 1
        out = self.order[self.cursor:self.cursor + k]
        self.cursor += k
        return [int(e) for e in out]

    # ── specialize: each centroid serves its own members ────────────────────
    def _centroid_batch(self, cid: int, k: int, standing) -> List[int]:
        members = standing.members(cid)
        if not members:                       # no home here yet: fall back to the sweep
            return self._sweep_batch(k)
        at = self.rota_at.get(cid, 0)
        rota = self.rota.get(cid)
        if rota != members:                   # membership moved: restart this rota
            self.rota[cid], rota, at = list(members), list(members), 0
        out = [rota[(at + i) % len(rota)] for i in range(min(k, len(rota)))]
        self.rota_at[cid] = (at + len(out)) % len(rota)
        return [int(e) for e in out]

    # ── the one entry point ─────────────────────────────────────────────────
    def experts(self, k: int, cid: int, standing, resident=()) -> List[int]:
        """The k experts that train on this input. In TEST that is the next
        group of the pool sweep, which ignores cid entirely — the point is that
        every expert meets every kind of material. In SPECIALIZE it is the next
        slice of the routed centroid's own rota."""
        k = max(1, int(k))
        self.rounds += 1
        out = (self._centroid_batch(int(cid), k, standing) if self.phase == SPECIALIZE
               else self._sweep_batch(k))
        # "he allocates first to already loaded experts" (Aman, p8). Order only —
        # WHICH experts run is the curriculum's business and is untouched; this
        # decides who runs FIRST, so the already-live adapter needs no swap at all
        # and the rest swap once each. Sorting, not filtering: dropping a swept
        # expert to avoid a swap would break equal exposure, which is the entire
        # reason the sweep exists.
        live = set(int(e) for e in resident)
        return sorted(out, key=lambda e: e not in live)

    def ready(self, standing) -> bool:
        """Testing is done: swept the floor and EVERY expert is rankable. The
        caller settles the general class on this, before migrating."""
        return (self.phase == TEST and self.sweeps >= MIN_SWEEPS
                and all(standing.rankable(e) for e in range(standing.E)))

    def advance(self, standing) -> bool:
        """Promote TEST -> SPECIALIZE once the pool has been swept enough for a
        mean to exist and migration has actually seated somebody. Returns True
        on the transition. Called after migrate(), never inside it."""
        if self.phase == SPECIALIZE or not standing.elite:
            return False
        if not any(standing.members(c) for c in range(standing.C)):
            return False                      # nothing was placed: stay in test
        self.phase = SPECIALIZE
        self.rota.clear()
        self.rota_at.clear()
        return True

    def state(self) -> Dict[str, object]:
        return {"phase": self.phase, "sweeps": self.sweeps, "rounds": self.rounds,
                "cursor": self.cursor, "E": self.E}

    def to_dict(self) -> Dict[str, object]:
        return {"E": self.E, "order": list(self.order), "cursor": self.cursor,
                "sweeps": self.sweeps, "rounds": self.rounds, "phase": self.phase}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Curriculum":
        c = Curriculum(int(d.get("E", C.E)))
        c.order = [int(x) for x in d.get("order", [])]
        c.cursor = int(d.get("cursor", 0))
        c.sweeps = int(d.get("sweeps", 0))
        c.rounds = int(d.get("rounds", 0))
        c.phase = str(d.get("phase", TEST))
        return c
