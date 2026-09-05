"""The loops.

  form      offline: gate hidden states over a corpus -> frozen geometry
  pretrain  Central alone, plain CE on real answers, over a token budget
  train     joint: route -> experts -> grounded score -> standing -> updates
  answer    deployment: same routing, Central synthesises, NOTHING is written

Training is dataset-based; deployment is input-based (Aman p.1). The live path
has no reward and no second scorer, so there is nothing to silently fall back to.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np

from . import config as C
from . import data, state
from .chain import MigrationChains, SizeChains
from .geometry import Geometry
from .health import Health
from .models import Central, ExpertPool, Gate, active_mb, measure_expert_peak_mb, peak_mb, reset_peak
from .reward import Reliability, is_heldout, score, score_one, weights
from .router import Plan, Router
from .scheduler import Scheduler
from .standing import Standing


class System:
    def __init__(self):
        self.gate = Gate()
        self.central = Central()
        self.pool = ExpertPool()
        self.geo: Optional[Geometry] = None
        self.standing: Optional[Standing] = None
        self.rel: Optional[Reliability] = None
        self.mig: Optional[MigrationChains] = None
        self.size: Optional[SizeChains] = None
        self.sched: Optional[Scheduler] = None
        self.router: Optional[Router] = None
        self.health = Health()
        self.clock = 0
        self.batch = 0
        self.consumed: Dict[str, int] = {}
        self._stream = None

    # ── boot ────────────────────────────────────────────────────────────────
    def boot(self, need_experts: bool = True) -> "System":
        data.authenticate()
        reset_peak()
        base = active_mb()
        self.gate.load()
        gate_mb = active_mb() - base
        self.central.load()
        central_mb = active_mb() - base - gate_mb
        # working reserve = Central's CE-forward peak (the logits spike), measured
        reset_peak()
        probe_y = self.central.encode("The quick brown fox jumps over the lazy dog. " * 12)[:C.TARGET_MAX_TOKENS]
        self.central.ce_vector(self.central.context_ids("probe", [], len(probe_y)), probe_y)
        working_mb = max(0.0, peak_mb() - active_mb())
        expert_mb = measure_expert_peak_mb() if need_experts else None
        self.sched = Scheduler(self.pool, expert_mb, central_mb, gate_mb, working_mb)
        self._restore()
        return self

    def _restore(self) -> None:
        blob = state.load()
        gate_hash = self.gate.weight_hash()
        if blob and blob.get("geometry"):
            geo = Geometry.from_dict(blob["geometry"])
            if geo.version.get("gate") != gate_hash:
                print(f"[state] geometry was formed with gate {geo.version.get('gate')} but the gate is "
                      f"{gate_hash} — geometry INVALID, re-form before training")
                geo = None
            self.geo = geo
        if self.geo is not None and blob:
            self.standing = Standing.from_dict(blob["standing"]) if blob.get("standing") else Standing(self.geo.C)
            self.rel = blob.get("reliability") or Reliability(self.geo.C)
            self.mig = blob.get("migration") or MigrationChains(self.geo.C)
            self.size = blob.get("size") or SizeChains(self.geo.C)
            self.clock = int(blob.get("clock", 0))
            self.batch = int(blob.get("batch", 0))
            self.consumed = dict(blob.get("consumed", {}))
            print(f"[state] restored: {self.geo.C} clusters, standing n={self.standing.total_n():.0f}, "
                  f"reliability obs={self.rel.total_obs():.0f}, clock={self.clock}, batch={self.batch}")
        self._wire()

    def _wire(self) -> None:
        if self.geo is None:
            return
        if self.standing is None:
            self.standing, self.rel = Standing(self.geo.C), Reliability(self.geo.C)
            self.mig, self.size = MigrationChains(self.geo.C), SizeChains(self.geo.C)
        self.router = Router(self.gate, self.geo, self.standing, self.sched)

    def save(self) -> None:
        if self.geo is None:
            return
        state.save({"geometry": self.geo.to_dict(), "standing": self.standing.to_dict(),
                    "reliability": self.rel, "migration": self.mig, "size": self.size,
                    "clock": self.clock, "batch": self.batch, "consumed": self.consumed})
        self.gate.save()
        for e in list(self.pool.resident):
            self.pool.save(e)

    def stream(self):
        if self._stream is None:
            self._stream = data.iter_mixture(consumed=self.consumed)
        return self._stream

    # ── form ────────────────────────────────────────────────────────────────
    def form(self, n_samples: int) -> Geometry:
        vecs = []
        for s in self.stream():
            ids = self.gate.encode(s.prompt)[:512]
            vecs.append(self.gate.hidden(ids).mean(0))
            if len(vecs) >= n_samples:
                break
        X = np.stack(vecs)
        self.geo = Geometry.form(X, C.MAX_CLUSTERS, {"gate": self.gate.weight_hash(),
                                                     "extractor": "extract_pair.v1"})
        self.standing = self.rel = self.mig = self.size = None
        self._wire()
        self.save()
        return self.geo

    # ── pretrain ────────────────────────────────────────────────────────────
    def pretrain(self, n_tokens: int) -> Dict[str, float]:
        done, losses, t0 = 0, [], time.time()
        for s in self.stream():
            loss, n = self.central.pretrain_step(s.prompt, s.answer)
            if math.isfinite(loss):
                losses.append(loss)
            done += n
            if len(losses) % 10 == 0 and losses:
                print(f"[pretrain] {done}/{n_tokens} tokens, ce {np.mean(losses[-10:]):.3f}")
            if done >= n_tokens:
                break
        self.central.save()
        return {"tokens": done, "ce_first": float(np.mean(losses[:10])) if losses else float("nan"),
                "ce_last": float(np.mean(losses[-10:])) if losses else float("nan"), "sec": time.time() - t0}

    # ── train ───────────────────────────────────────────────────────────────
    def train(self, n_batches: int) -> Dict[str, float]:
        if self.geo is None:
            raise RuntimeError("no geometry: run `form` first")
        t0, admitted, skipped = time.time(), 0, 0
        target = self.batch + int(n_batches)                 # N MORE batches, not an absolute index
        n0, r0 = self.standing.total_n(), self.rel.total_obs()
        stream = self.stream()
        while self.batch < target:                    # check first: never pull a row we will not use
            s = next(stream, None)
            if s is None:
                break
            r = self._train_one(s)
            admitted += int(r.get("admitted", 0))
            skipped += int(not r.get("admitted", 0))
            self.batch += 1
            if self.batch % C.MIGRATE_EVERY == 0:
                moves = self.standing.migrate(self.mig)
                if moves:
                    print(f"[migrate] b{self.batch}: {len(moves)} moves {moves[:6]}")
            if self.batch % C.SAVE_EVERY == 0:
                self.save()
        self.save()
        return {"batches": self.batch, "ran": self.batch - (target - int(n_batches)), "admitted": admitted,
                "skipped": skipped, "sec": time.time() - t0,
                "standing_n": self.standing.total_n(), "reliability_obs": self.rel.total_obs(),
                "measured_this_run": (self.standing.total_n() - n0) + (self.rel.total_obs() - r0)}

    def _train_one(self, s: data.Sample) -> Dict[str, float]:
        plan = self.router.plan(s.prompt)
        resident = self.sched.ensure([sel.eid for sel in plan.selections])
        sels = [sel for sel in plan.selections if sel.eid in resident]
        y = self.central.target_ids(s.answer)
        assign_y = self.geo.assign_tokens(self.gate.hidden(self.gate.encode(s.answer)[:512]))
        texts: Dict[int, str] = {}
        spans: Dict[int, str] = {}
        for sel in sels:
            span_text = self.gate.tok.decode(plan.ids[sel.start:sel.end])
            spans[sel.eid] = span_text
            texts[sel.eid], _ = self.pool.run(sel.eid, span_text, s.prompt)
        sc = score(self.central, s.prompt, y, assign_y, texts, self.rel)
        heldout = is_heldout(s.key)
        self.health.deltas_seen(sc.deltas, sc.zero_delta)
        self.health.texts_seen(texts)
        rec: Dict[str, float] = {"k": len(sels), "k_wanted": plan.k_wanted, "T": len(plan.ids), "M": len(y),
                                 "rho": sc.rho, "heldout": float(heldout), "admitted": float(sc.admitted)}
        if heldout:
            self.rel.observe(sc.b, assign_y)
        elif sc.admitted:
            for sel in sels:
                self.standing.observe(sel.eid, sel.cid, sc.deltas[sel.eid], sel.n_tokens)
            rec["gate_loss"] = self.router.gate_step(plan, sc.deltas)
            rec["expert_loss"] = self._expert_update(sels[0], spans, texts, sc, s, y, assign_y) if sels else 0.0
        self._breathe(sels, len(plan.ids))
        self.clock += len(plan.ids)
        self.health.put(**rec, clock=self.clock, active_mb=active_mb())
        self.health.tick(self.standing.total_n(), self.rel, self.size, self.mig)
        d = " ".join(f"e{sel.eid}{'*' if sel.trial else ''}:{sc.deltas.get(sel.eid, float('nan')):+.3f}" for sel in sels)
        losses = " ".join(f"{k}={rec[k]:.3f}" for k in ("gate_loss", "expert_loss") if k in rec)
        print(f"[b{self.batch}] {s.source} k={len(sels)}/{plan.k_wanted} home={plan.home} present={plan.present} "
              f"rho={sc.rho:.2f} {'HELDOUT' if heldout else ('ADMIT' if sc.admitted else 'REFUSE')} {d} {losses}")
        return rec

    def _breathe(self, sels, T: int) -> None:
        """Territory: tau follows PREDICTED load; direction never moves. Load is
        the share of tokens actually allocated to each cluster — the quantity tau
        affects — so tightening reduces observed load and the loop self-corrects.
        Measuring composition weight instead would ignore tau and run away."""
        share = np.zeros(self.geo.C)
        for sel in sels:
            share[sel.cid] += sel.n_tokens
        share /= max(1, T)
        for c in range(self.geo.C):
            self.size.observe_load(c, float(share[c]), self.geo.C)
            self.geo.set_tau(c, self.size.next_tau(c, float(self.geo.tau[c])))

    def _expert_update(self, sel, spans, texts, sc, s, y, assign_y) -> float:
        """Reward-weighted self-imitation on the home expert: a second sampled
        candidate, standardised advantage, CE on its own text signed by which
        one helped Central more. Silences itself when both score alike."""
        eid = sel.eid
        greedy, d_g = texts[eid], sc.deltas[eid]
        sampled = self.pool.sample(eid, spans[eid], s.prompt)
        if not sampled or sampled == greedy:
            return 0.0
        w = weights(self.rel, assign_y, len(sc.b))
        d_s = score_one(self.central, s.prompt, y, sc.b, w, sampled)
        sd = abs(d_g - d_s)
        if sd < 1e-6:
            return 0.0
        m = (d_g + d_s) / 2.0
        adv = [(d_g - m) / sd, (d_s - m) / sd]
        return self.pool.update(eid, self.pool.prompt(spans[eid], s.prompt), [greedy, sampled], adv)

    # ── answer (deployment; writes nothing) ─────────────────────────────────
    def answer(self, prompt: str, max_tokens: int = 256) -> Dict[str, object]:
        if self.geo is None:
            return {"text": self.central.generate(prompt, [], max_tokens), "k": 0, "trust": 1.0, "notes": []}
        plan = self.router.plan(prompt)
        resident = self.sched.ensure([sel.eid for sel in plan.selections])
        notes: List[tuple] = []
        for sel in plan.selections:
            if sel.eid not in resident:
                continue
            span_text = self.gate.tok.decode(plan.ids[sel.start:sel.end])
            text, _ = self.pool.run(sel.eid, span_text, prompt)
            st = self.standing.score(sel.eid, sel.cid)
            notes.append((st if st is not None else -1e9, text))
        self._breathe([sel for sel in plan.selections if sel.eid in resident], len(plan.ids))
        self.clock += len(plan.ids)
        trust = float(plan.w @ self.rel.vector())
        m = max(1, int(math.ceil(len(notes) * (1.0 - trust)))) if notes else 0
        notes.sort(key=lambda x: -x[0])
        use = [t for _, t in notes[:m]]
        return {"text": self.central.generate(prompt, use, max_tokens), "k": len(notes), "trust": trust,
                "notes": use, "home": plan.home, "w": plan.w.round(3).tolist()}
