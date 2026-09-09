"""One health record, one printer, the canaries.

The old system had NO stage whose job was to ask "is this mechanism doing
anything?" — which is how a fitted curve emitted 32 forever. Every mechanism
here publishes into this record and it is printed every HEALTH_EVERY batches
(rule 22). Every scalar is finite-checked at the point it is RECORDED.

Canaries (see docs/grounded-reward.md):
  A  standing counters advance within STANDING_A_WINDOW batches   (y never arrives / all refused)
  B  zero-delta fraction; dropped expert texts                     (both forwards saw the same context)
  C  rolling std of delta; sign fraction 15-85%                    (delta degenerates to a band)
  E  reliability vector flatness                                   (vector collapses to the scalar)
  U  expert updates actually applied vs skipped                    (a loss printed for a step that never ran)
Canary D (verifiable exact-match split) is NOT implemented: it needs a Central
generation per batch. Open.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List

import numpy as np

from . import config as C


class Health:
    def __init__(self):
        self.batch = 0
        self.rec: Dict[str, float] = {}
        self.deltas = deque(maxlen=200)
        self.zero = deque(maxlen=100)
        self.n_hist = deque(maxlen=C.STANDING_A_WINDOW)
        self.clones = deque(maxlen=100)
        self.applied = deque(maxlen=100)      # 1 = an expert update ran, 0 = skipped
        self.updates = 0
        self.dropped = 0
        self.alarms: List[str] = []
        self.nonfinite = 0

    def put(self, **kv) -> None:
        for k, v in kv.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                if not math.isfinite(float(v)):
                    self.nonfinite += 1
                    self.alarms.append(f"NONFINITE {k}")
                    continue
                self.rec[k] = float(v)
            else:
                self.rec[k] = v

    def texts_seen(self, texts: Dict[int, str]) -> None:
        """Clone rate: experts are LoRA over one base and start identical, so at
        cold start they say the same thing. This must FALL as training diverges
        them; if it does not, the expert update is not doing anything."""
        if len(texts) >= 2:
            vals = list(texts.values())
            self.clones.append(1.0 if len(set(vals)) < len(vals) else 0.0)

    def deltas_seen(self, deltas: Dict[int, float], zero: Dict[int, bool], dropped: int = 0) -> None:
        for e, d in deltas.items():
            if math.isfinite(d):
                self.deltas.append(d)
            self.zero.append(1.0 if zero.get(e, False) else 0.0)
        self.dropped += int(dropped)

    def update_seen(self, loss) -> None:
        """None = skipped (nothing ran); a float = an optimiser step happened."""
        if loss is None:
            self.applied.append(0.0)
        else:
            self.applied.append(1.0)
            self.updates += 1

    def tick(self, standing_total_n: float, reliability, size_chains=None, migration=None) -> List[str]:
        self.batch += 1
        alarms: List[str] = []
        self.n_hist.append(standing_total_n)
        if len(self.n_hist) == self.n_hist.maxlen and self.n_hist[-1] <= self.n_hist[0]:
            alarms.append(f"A: standing has not advanced in {self.n_hist.maxlen} batches — y is not arriving or every batch is refused")
        if len(self.zero) >= 50 and float(np.mean(self.zero)) > 0.05:
            alarms.append(f"B: {100*float(np.mean(self.zero)):.0f}% of expert deltas identically zero — same context both passes?")
        if len(self.deltas) >= 50:
            arr = np.array(self.deltas)
            sd, neg = float(arr.std()), float((arr < 0).mean())
            self.put(delta_std=sd, delta_neg_frac=neg, delta_mean=float(arr.mean()))
            if sd < 0.01:
                alarms.append(f"C: delta std {sd:.4f} < 0.01 — reward degenerating to a band")
            if not (0.15 <= neg <= 0.85):
                alarms.append(f"C: {100*neg:.0f}% of deltas negative — a scorer that cannot say 'hurt' is not measuring")
        if self.clones:
            self.put(clone_frac=float(np.mean(self.clones)))
        if len(self.applied) >= 50:
            frac = float(np.mean(self.applied))
            self.put(update_frac=frac)
            if frac < 0.5:
                alarms.append(f"U: only {100*frac:.0f}% of admitted batches applied an expert update")
        self.put(expert_updates=self.updates, dropped_texts=self.dropped)
        flat = reliability.flatness()
        if flat is not None:
            self.put(reliability_spread=flat)
            if flat < 0.05:
                alarms.append(f"E: reliability spread {flat:.3f} — vector has flattened to a scalar")
        self.put(reliability_obs=reliability.total_obs(), standing_n=standing_total_n)
        if size_chains is not None:
            acc = size_chains.accuracy()
            self.put(size_chain_acc=(acc if acc is not None else -1.0))
        if migration is not None:
            acc = migration.pool.accuracy()
            self.put(migration_chain_acc=(acc if acc is not None else -1.0))
        self.alarms.extend(alarms)
        if self.batch % C.HEALTH_EVERY == 0:
            self.print()
        return alarms

    def print(self) -> None:
        parts = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(self.rec.items())]
        print(f"[health b{self.batch}] " + " ".join(parts))
        if self.alarms:
            for a in self.alarms[-5:]:
                print(f"[health] ALARM {a}")
