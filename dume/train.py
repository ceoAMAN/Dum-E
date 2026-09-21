"""The loops.

  form      offline: gate hidden states over a corpus -> frozen geometry
  pretrain  Central alone, plain CE on real answers, over a token budget
  train     joint: route -> experts -> grounded score -> standing -> updates
  answer    deployment: same routing, Central synthesises, NOTHING is written
            (no reward, no standing, no tau, no chains, no clock)

Training is dataset-based; deployment is input-based (Aman p.1). The live path
has no reward and no second scorer, so there is nothing to silently fall back to.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from . import data, state
from .alloc import AllocLaw
from .chain import MigrationChains, SizeChains, ThermalRegulator
from .curriculum import Curriculum
from .geometry import Geometry
from .health import Health
from .models import (Central, ExpertPool, Gate, active_mb, measure_expert_peak_mb, thermal_state,
                     measure_update_slope_mb, measure_slot_mb, reset_peak)
from .reward import CentralBand, Reliability, is_heldout, rho_of, score, score_one
from .router import Router
from .scheduler import Scheduler
from .standing import Standing


def _restored(obj, cls, *args):
    """Unpickle only if the stored object still matches the CURRENT class.

    state.VERSION guards the blob's SHAPE, but these values are pickled live
    objects: unpickling restores an old instance's __dict__ into the new class,
    so a field added since the checkpoint is simply absent and the first read of
    it raises. That is what happened to CentralBand when the quantile band was
    replaced by the gain gate — the store looked fine and boot would have failed
    on the next attribute access. A mismatched object is rebuilt cold rather
    than half-restored; only that mechanism's history is lost, never the
    geometry or the standing.
    """
    if obj is None or not isinstance(obj, cls):
        return cls(*args)
    missing = [k for k in vars(cls(*args)) if not hasattr(obj, k)]
    if missing:
        print(f"[state] {cls.__name__} predates {', '.join(missing)} — rebuilding it cold")
        return cls(*args)
    return obj


def orientation(summary: str, i: int, n: int, start: int, end: int, T: int) -> str:
    """What an expert is told about the input it cannot see.

    Two parts, and NEITHER is a copy of anyone's span. `summary` is the gate's
    bounded map of the whole input, shared by the batch. The position line is
    free — the router already knows it — and it is the only part that DIFFERS
    per expert, which is what the old prompt never supplied: it re-printed the
    excerpt with no indication of where in the input it sat.

    Position is stated in the same units apex-nadir allocates in, so an expert
    can tell a 12-token slice of 51 from a 12-token slice of 2000."""
    where = f"Section {i + 1} of {n}, tokens {start}-{end} of {T}."
    return f"{summary}\n{where}" if summary else where


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
        self.alloc: Optional[AllocLaw] = None
        self.band: Optional[CentralBand] = None
        self.curric: Optional[Curriculum] = None
        self.therm: Optional[ThermalRegulator] = None
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
        # no working-memory probe: the reserve is sqrt(R), not a measured spike, so
        # a full 1024-token Central forward at every boot would cost a real memory
        # spike to produce a number nothing reads.
        expert_mb = measure_expert_peak_mb() if need_experts else None
        slope = slot = 0.0
        if need_experts:
            self.pool.load(0)                       # a real expert, wrapped exactly as it runs
            slope = measure_update_slope_mb(self.pool, 0)
            slot = measure_slot_mb(self.pool)
            self.pool.unload(0)
        self.sched = Scheduler(self.pool, expert_mb, central_mb, gate_mb, slope, slot)
        self.central.version = self.pool.version = self.gate.weight_hash()
        self._restore()
        return self

    def _restore(self) -> None:
        blob = state.load()
        gate_hash = self.gate.weight_hash()
        if blob:            # the clock and the stream position are independent of the geometry
            self.clock = int(blob.get("clock", 0))
            self.batch = int(blob.get("batch", 0))
            self.consumed = dict(blob.get("consumed", {}))
        if blob and blob.get("geometry"):
            geo = Geometry.from_dict(blob["geometry"])
            if geo.version.get("gate") != gate_hash:
                print(f"[state] geometry was formed with gate {geo.version.get('gate')} but the gate is "
                      f"{gate_hash} — geometry INVALID, re-form before training")
                geo = None
            self.geo = geo
        if self.geo is not None and blob:
            self.standing = Standing.from_dict(blob["standing"]) if blob.get("standing") else Standing(self.geo.C)
            self.rel = _restored(blob.get("reliability"), Reliability, self.geo.C)
            self.mig = _restored(blob.get("migration"), MigrationChains, self.geo.C)
            self.size = _restored(blob.get("size"), SizeChains, self.geo.C)
            self.alloc = _restored(blob.get("alloc"), AllocLaw)
            self.band = _restored(blob.get("band"), CentralBand)
            self.therm = _restored(blob.get("thermal"), ThermalRegulator)
            if blob.get("curriculum"):
                self.curric = Curriculum.from_dict(blob["curriculum"])
            print(f"[state] restored: {self.geo.C} clusters, standing n={self.standing.total_n():.0f}, "
                  f"reliability obs={self.rel.total_obs():.0f}, clock={self.clock}, batch={self.batch}")
        self._wire()

    def _wire(self) -> None:
        if self.geo is None:
            return
        if self.standing is None:
            self.standing, self.rel = Standing(self.geo.C), Reliability(self.geo.C)
            self.mig, self.size = MigrationChains(self.geo.C), SizeChains(self.geo.C)
        if self.alloc is None:
            self.alloc = AllocLaw()
        if self.band is None:
            self.band = CentralBand()
        if self.curric is None:
            self.curric = Curriculum()
        if self.therm is None:
            self.therm = ThermalRegulator()
        self.sched.thermal = self.therm
        self.router = Router(self.gate, self.geo, self.standing, self.sched, self.alloc)

    def save(self) -> None:
        if self.geo is None:
            return
        state.save({"geometry": self.geo.to_dict(), "standing": self.standing.to_dict(),
                    "reliability": self.rel, "migration": self.mig, "size": self.size,
                    "alloc": self.alloc, "band": self.band, "curriculum": self.curric.to_dict(),
                    "thermal": self.therm,
                    "clock": self.clock, "batch": self.batch, "consumed": self.consumed})
        self.gate.save()
        self.pool.save_all()
        # Central trains in-loop now; before this line every one of its steps
        # was discarded at exit and Central.load() restored the pretrain ckpt.
        self.central.save()

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
        self.standing = self.rel = self.mig = self.size = self.alloc = self.band = None
        self.therm = None
        self.curric = None                   # a new geometry retires the schedule too
        self.batch, self.clock = 0, 0        # a new geometry has no history; `consumed` stays
        self.health = Health()
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
            if losses and len(losses) % 10 == 0:
                print(f"[pretrain] {done}/{n_tokens} tokens, ce {np.mean(losses[-10:]):.3f}")
            if done >= n_tokens:
                break
            if losses and len(losses) % C.SAVE_EVERY == 0:
                self.central.save()
                self.save()                  # persist `consumed`: never re-feed train the pretrain rows
        self.central.save()
        self.save()
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
            admitted += int(r.get("graded", 0))
            skipped += int(not r.get("graded", 0))
            self.batch += 1
            if self.batch % C.MIGRATE_EVERY == 0:
                if self.curric.ready(self.standing):
                    gen = self.standing.settle_generals(self.geo.domain())
                    print(f"[curriculum] testing done: general class locked {gen}")
                moves = self.standing.migrate(self.mig, cap=self.geo.capacity(), domain=self.geo.domain())
                if moves:
                    print(f"[migrate] b{self.batch}: {len(moves)} moves {moves[:6]}")
                if self.curric.advance(self.standing):
                    seated = sum(len(self.standing.members(c)) for c in range(self.geo.C))
                    print(f"[curriculum] b{self.batch}: TEST -> SPECIALIZE after "
                          f"{self.curric.sweeps} sweeps; {seated} experts seated")
            if self.batch % C.SAVE_EVERY == 0:
                self.save()
        self.save()
        return {"batches": self.batch, "ran": self.batch - (target - int(n_batches)), "graded": admitted,
                "not_graded": skipped, "sec": time.time() - t0,
                "standing_n": self.standing.total_n(), "reliability_obs": self.rel.total_obs(),
                "timeline_a_rate": self.band.rate(), "graded_this_run": self.standing.total_n() - n0,
                "reliability_this_run": self.rel.total_obs() - r0}

    def _split(self, plan) -> List[Tuple[object, str, str]]:
        """The gate's whole job on an input: cut it into spans and write each
        expert's prompt material. Runs BEFORE any expert is made resident, and
        reads only `plan`, so nothing decided AFTER the plan — which experts
        ensure() managed to seat, what it had to evict — can change a prompt.

        That is a narrower guarantee than "a function of the input", and the
        difference is worth stating because the earlier comment here overclaimed
        it. `plan` already carries residency and temperature: curriculum.experts()
        orders its picks resident-first (curriculum.py:136) so WHICH expert gets
        WHICH span follows what was already loaded, and k is thermally regulated
        so the span sizes — hence this map's budget — move with the chassis.

        THE MAP'S BUDGET IS APEX-NADIR'S SMALLEST ALLOCATION (Aman, 2026-09-21):
        deciding how many tokens an expert processes is what the law is for, so
        the map of the whole obeys it too. The MINIMUM span rather than the mean,
        so one summary per batch is small enough for every seat at the table.

        Returns a LIST parallel to plan.selections, never a dict keyed by eid.
        Deployment emits `cycles` fragments per expert — several Selections that
        share an eid and differ only in [start, end) — so an eid-keyed dict keeps
        one span per expert and hands every fragment the last one's text. The
        fragments are the point: they are how an expert covers a region larger
        than its span without paying a swap."""
        sels = list(plan.selections)
        summary = self.gate.summarise(plan.ids, min((x.n_tokens for x in sels), default=0))
        return [(sel, self.gate.tok.decode(plan.ids[sel.start:sel.end]),
                 orientation(summary, i, len(sels), sel.start, sel.end, len(plan.ids)))
                for i, sel in enumerate(sels)]

    def _train_one(self, s: data.Sample) -> Dict[str, float]:
        # training PROBES: spans come from {t_lo, t_mid, t_hi} rather than the
        # fitted allocation, so span size is explored at no extra cost. Only the
        # 2*sqrt(E) extreme-ranked experts (§2) feed the curves.
        plan = self.router.plan(s.prompt, probe=True, curriculum=self.curric)
        y = self.central.target_ids(s.answer)
        # token types are read off the TARGET IDS THEMSELVES, so assign_y[t] pairs with
        # b[t] by construction — the reliability writer and the weight reader index the
        # same t. (Resampling a differently-sized assignment silently mismatched them.)
        assign_y = self.geo.assign_tokens(self.gate.hidden(y))
        texts: Dict[int, str] = {}
        emitted: Dict[int, int] = {}
        secs: Dict[int, float] = {}
        # THE GATE FINISHES BEFORE THE POOL STARTS (Aman, 2026-09-21): split the input
        # into prompts, THEN activate k and move them to processing. Two reasons, and
        # the second is the load-bearing one:
        #
        #   ORDER — the gate's whole job finishes before the pool's begins, so a
        #   prompt cannot be changed by anything ensure() does. Budgeting the map off
        #   `sels` made it depend on ensure()'s OUTCOME: the same row with a different
        #   seat count got a different budget and a different map. What the law
        #   allocated is what the law allocated.
        #
        #   RAM — weaker than it looks, and stated here so nobody relies on it.
        #   summarise() peaks at 959 MB on a long prefill (measured), but residency is
        #   NOT batch-scoped: unload() is only called from inside ensure()
        #   (scheduler.py:121), so at this point the previous batch's adapters are
        #   still loaded. The peak is essentially unchanged; only the first batch of a
        #   run sees the gate alone.
        cut = self._split(plan)
        # probe => cycles == 1 => exactly one Selection per eid, so keying the
        # training maps by eid is exact here. Deployment is NOT: it iterates `cut`.
        spans = {sel.eid: sp for sel, sp, _ in cut}
        ctxs = {sel.eid: cx for sel, _, cx in cut}
        resident = self.sched.ensure([sel.eid for sel in plan.selections])
        sels = [sel for sel in plan.selections if sel.eid in resident]
        share = self._context_share(len(sels), s.prompt, len(y))
        for sel in sels:
            t_run = time.time()
            budget = self.alloc.budget(sel.n_tokens, sel.n_tokens, share)
            texts[sel.eid], _ = self.pool.run(sel.eid, spans[sel.eid], ctxs[sel.eid], budget=budget)
            secs[sel.eid] = time.time() - t_run
            emitted[sel.eid] = len(self.pool.tok.encode(texts[sel.eid])) if texts[sel.eid] else 0
        sc = score(self.central, s.prompt, y, plan.w, texts, self.rel)
        heldout = is_heldout(s.key) and not s.self_referent
        self.health.deltas_seen(sc.deltas, sc.zero_delta, dropped=len(sc.dropped))
        self.health.texts_seen(texts)
        trust = rho_of(self.rel, plan.w)     # Central's score on THIS input's composition
        place = AllocLaw.placement(self.standing.rank(self.geo.domain()))
        u_best = max([place.get(sel.eid, 0.5) for sel in sels], default=0.5)
        tl = self.band.decide(self.alloc, len(plan.ids), u_best)
        rec: Dict[str, float] = {"k": len(sels), "k_wanted": plan.k_wanted, "T": len(plan.ids), "M": len(y),
                                 "rho": sc.rho, "heldout": float(heldout), "admitted": float(sc.admitted),
                                 "trust": trust, "timeline_a": self.band.rate()}
        if self.band.last_gain is not None:
            rec["pred_gain"] = float(self.band.last_gain)
        wv = self.alloc.width_varies()
        if wv is not None:
            rec["alloc_width_var"] = wv
        if heldout:
            self.rel.observe(sc.b, assign_y)
        elif sc.admitted:
            for sel in sels:
                if sel.eid not in sc.deltas:        # its text never reached Central; nothing was measured
                    continue
                rec["graded"] = 1.0
                # divisor = tokens the expert EMITTED, never the span the router handed it
                self.standing.observe(sel.eid, sel.cid, sc.deltas[sel.eid], emitted[sel.eid],
                                      sc.correct.get(sel.eid, 0.0), sc.halluc.get(sel.eid, 0.0),
                                      secs.get(sel.eid, 0.0))
            # only the 2*sqrt(E) extreme-ranked experts feed the curves (§2): every
            # expert is ALLOCATED per-expert, but fitting off the extremes is what
            # saves the compute and gives the envelope its spread.
            probe_set = set(self.alloc.sample(self.standing.rank(self.geo.domain())))
            for sel in sels:
                if sel.eid in sc.deltas and sel.eid in probe_set:
                    self.alloc.observe(sel.n_tokens, sc.deltas[sel.eid], secs.get(sel.eid))
            rec["gate_loss"] = self.router.gate_step(plan, sc.deltas)
            loss = self._expert_update(self._update_target(plan, sels, sc), spans, ctxs, texts, sc, s, y)
            self.health.update_seen(loss)
            if loss is not None:
                rec["expert_loss"] = loss
            imit = self._imitate(sels, spans, ctxs, texts, sc, s)
            if imit is not None:
                rec["imitate_loss"] = imit
        # Central trains on the real answer, at FULL weight, on EVERY sample that
        # has one. Training has y, so there is nothing to discount: "natural
        # gradients will flow" (Aman, 2026-09-17). The reliability score rho is
        # MEASURED here (Reliability.observe, held-out shard) and APPLIED only in
        # deployment, where there is no y and `trust` decides how much of the
        # synthesis Central holds — see answer(). The scalar rho on the training
        # gradients was a deduction applied where nothing needed deducting.
        #
        # Deliberately OUTSIDE `elif sc.admitted`: admission (rho >= R_MIN and
        # M >= TARGET_MIN_TOKENS) is support for the expert DELTA — a mean over M
        # tokens needs M, and a composition Central cannot read cannot grade an
        # expert. Neither reason applies to Central's own CE on y: a 2-token
        # answer is still a real answer, and rho < R_MIN is exactly where Central
        # most needs the data. Nesting it under admission re-imposed a rho gate.
        #
        # Two samples are excluded, both because y is not REAL there:
        #   held-out      reliability is fitted on it; training on it would
        #                 flatter the number deployment reads.
        #   self_referent y is Central's OWN delivered answer (dead_time). A
        #                 legitimate frozen referent for scoring EXPERTS against,
        #                 but Central learning from it is self-distillation with
        #                 no ground, at full weight. Note heldout is already
        #                 False for these, so the guard must be explicit.
        #
        # LAST in the batch, not earlier: _expert_update re-runs Central through
        # score_one to grade the sampled candidate against the SAME baseline b
        # measured at the top. Stepping Central first would grade d_s on a
        # different instrument than b, and the paired difference is only
        # meaningful while Central is frozen — the premise of the reward.
        if not heldout and not s.self_referent and self.central.model is not None:
            cl, _ = self.central.pretrain_step(s.prompt, s.answer)
            if math.isfinite(cl):
                rec["central_loss"] = cl
        if self.alloc.input_seen(len(plan.ids)):          # every E inputs
            self.alloc.fit()
            print(f"[alloc] {self.alloc.state()}")
        self._breathe(plan)
        self.clock += len(plan.ids)
        # the device's side of the k tug of war, recorded BEFORE the put so a
        # run can show whether heat ever got a vote at all
        rec["k_thermal"] = float(self.sched.k_thermal)   # observes the level itself
        ts = self.therm.state()
        rec["thermal"] = ts["last"]
        rec["thermal_base"] = ts["baseline"]
        rec["thermal_vol"] = ts["volatility"]
        rec["thermal_peak"] = ts["peak"]
        self.health.put(**rec, clock=self.clock, active_mb=active_mb())
        self.health.tick(self.standing.total_n(), self.rel, self.size, self.mig)
        d = " ".join(f"e{sel.eid}{'*' if sel.trial else ''}:{sc.deltas.get(sel.eid, float('nan')):+.3f}" for sel in sels)
        losses = " ".join(f"{k}={rec[k]:.3f}" for k in ("gate_loss", "expert_loss", "central_loss") if k in rec)
        print(f"[b{self.batch}] {s.source} T{tl} k={len(sels)}/{plan.k_wanted} home={plan.home} present={plan.present} "
              f"M={len(y)} rho={sc.rho:.2f} {'HELDOUT' if heldout else ('ADMIT' if sc.admitted else 'REFUSE')} {d} {losses}")
        return rec

    def _context_share(self, k: int, question: str, n_target: int) -> int:
        """Each expert's slice of what is actually left in Central's window once
        the question and the whole of y are accounted for. MEASURED off the real
        limit — this is the constraint that genuinely binds on a 16 GB machine."""
        free = self.central.limit() - n_target - min(len(self.central.encode(question)),
                                                     C.WORKING_PROBE_TOKENS)
        return max(C.EXPERT_GEN_TOKENS, int(free) // max(1, int(k)))

    @staticmethod
    def _update_target(plan, sels, sc):
        """The HOME expert learns, not whichever span happened to come first in the
        document. Falls back to the best-scoring non-trial seat, then to any seat."""
        scored = [sel for sel in sels if sel.eid in sc.deltas]
        if not scored:
            return None
        home = [sel for sel in scored if sel.cid == plan.home and not sel.trial]
        body = [sel for sel in scored if not sel.trial]
        return (home or body or scored)[0]

    def _breathe(self, plan) -> None:
        """Territory: tau follows PREDICTED presence; direction never moves.

        Load is how often a cluster's territory CONTAINS the input — the exact
        quantity tau gates, so tightening lowers it and the loop self-corrects —
        measured over a rolling window and scored against the population average,
        which is a reachable setpoint. The previous version measured each
        cluster's share of allocated tokens against 1/C; with at most k_max seats
        among C clusters no cluster could ever read HEALTHY, so every tau wound
        to a rail and territory became vacuous."""
        self.geo.grow(plan.inside)          # centroids EARN seats from the input they receive
        rate = self.size.observe(plan.inside)
        if rate is None:                    # window not full: no measurement, no move
            return
        for c in range(self.geo.C):
            self.geo.set_tau(c, self.size.next_tau(c, float(self.geo.tau[c])))

    def _expert_update(self, sel, spans, ctxs, texts, sc, s, y) -> Optional[float]:
        """Reward-weighted self-imitation on the home expert: a second sampled
        candidate, advantage in the reward's own units, CE on its own text signed
        by which one helped Central more. Returns None when nothing ran, so a
        skipped step is never reported as a loss of 0.000."""
        if sel is None:
            return None
        eid = sel.eid
        greedy, d_g = texts[eid], sc.deltas[eid]
        sampled = self.pool.sample(eid, spans[eid], ctxs[eid],
                                   budget=len(self.pool.tok.encode(greedy)) or None)
        if not sampled or sampled == greedy:
            return None
        d_s = score_one(self.central, s.prompt, y, sc.b, sampled, sc.base_len)
        if d_s is None:
            return None
        # Gate on the reward's OWN measured noise, not an invented 1e-6: standardising
        # by the pair's spread turned every difference, however tiny, into a full-strength
        # +-0.5 coin flip. Below the noise floor there is no direction to learn.
        floor = float(self.health.rec.get("delta_mad", 0.0)) * 0.25
        if abs(d_g - d_s) <= floor:
            return None
        m = (d_g + d_s) / 2.0
        # No rho on the advantage, scalar or per-token: d is a plain mean over
        # real y, and rho is a deployment deduction, not a training one. The
        # delta already carries reliability on its own (see reward.py header).
        return self.pool.update(eid, self.pool.prompt(spans[eid], ctxs[eid]), [greedy, sampled],
                                [d_g - m, d_s - m])

    def _imitate(self, sels, spans, ctxs, texts, sc, s) -> Optional[float]:
        """Dormant distillation. The trial seat learns the TEXT of the best-scoring
        seated expert on the same input: idle capacity improving on somebody
        else's gradient, which is the only way a dormant expert climbs out of
        general on scraps.

        The spec's weighted sum over superior experts degenerates to the argmax
        here — with k <= 4 there is at most one clearly-superior seat, and CE
        against the winner's text reuses pool.update rather than needing a new
        hidden-state path.
        # ponytail: single teacher; weight the top few if k ever gets large.
        """
        trial = next((x for x in sels if x.trial and x.eid in spans), None)
        if trial is None:
            return None
        peers = [x for x in sels if not x.trial and x.eid in sc.deltas and texts.get(x.eid)]
        if not peers:
            return None
        best = max(peers, key=lambda x: sc.deltas[x.eid])
        # The teacher must have HELPED, not merely have hurt less. d is measured
        # against y, which came off disk, so "everything in training is checked
        # against real data" (Aman, 2026-09-20) applies here too: a negative
        # delta is real data saying that note raised Central's CE, and pulling a
        # dormant expert toward it at a full +1.0 advantage teaches the damage.
        # The old guard was relative only, and 359 of the 470 imitation steps in
        # the last run (76.4%) had a teacher whose own delta was negative.
        if sc.deltas[best.eid] <= 0.0:
            return None                       # real data says this note hurt
        if sc.deltas[best.eid] <= sc.deltas.get(trial.eid, -1e9):
            return None                       # nothing superior to imitate
        return self.pool.update(trial.eid, self.pool.prompt(spans[trial.eid], ctxs[trial.eid]),
                                [texts[best.eid]], [1.0])

    def dead_time(self, prompt: str, delivered: str) -> Dict[str, float]:
        """Timeline B, run in the dead time AFTER Timeline A has served the user.

        A answered alone and that answer is already out the door — nothing an
        expert says can move it. So it is a legitimate frozen referent: B re-runs
        the same input WITH experts, scores each against the delivered text and
        trains. This is the only thing that keeps experts improving on deployment
        traffic, where no y ever arrives.

        Marked self_referent, so it never feeds Central's RELIABILITY: y is
        Central's own output and Central would grade itself perfectly by
        construction."""
        if self.geo is None or not delivered.strip():
            return {}
        r = self._train_one(data.Sample(source="deadtime", prompt=prompt,
                                        answer=delivered, self_referent=True))
        self.batch += 1
        return r

    # ── answer (deployment; writes nothing until dead_time) ─────────────────
    def answer(self, prompt: str, max_tokens: int = 256) -> Dict[str, object]:
        if self.geo is None:
            return {"text": self.central.generate(prompt, [], max_tokens), "k": 0, "timeline": "A",
                    "trust": 1.0, "notes": []}
        plan = self.router.plan(prompt, probe=False)     # fitted allocation, padded spans
        trust = rho_of(self.rel, plan.w)     # Central's score on THIS input's composition
        place = AllocLaw.placement(self.standing.rank(self.geo.domain()))
        u_best = max([place.get(sel.eid, 0.5) for sel in plan.selections], default=0.5)
        if self.band.decide(self.alloc, len(plan.ids), u_best) == "A":
            # Timeline A: apex-nadir predicts the allocation buys nothing here, so
            # Central answers alone. K=0, no expert is loaded, and nothing is
            # written — deployment has no y, so nothing here could be grounded.
            return {"text": self.central.generate(prompt, [], max_tokens), "k": 0, "timeline": "A",
                    "trust": trust, "notes": [], "home": plan.home, "w": plan.w.round(3).tolist()}
        # same order as training: the gate splits, THEN k are activated. An expert
        # that trained with a map and a position must not meet a bare span in
        # production, and the split must not depend on residency in either path.
        cut = self._split(plan)
        resident = self.sched.ensure([sel.eid for sel in plan.selections])
        share = self._context_share(len(plan.selections), prompt, max_tokens)
        notes: List[tuple] = []
        for sel, span_text, ctx in cut:            # per FRAGMENT, not per expert
            if sel.eid not in resident:
                continue
            text, _ = self.pool.run(sel.eid, span_text, ctx,
                                    budget=self.alloc.budget(sel.n_tokens, sel.n_tokens, share))
            st = self.standing.score(sel.eid, sel.cid)
            notes.append((st if st is not None else -1e9, text))
        # no _breathe, no clock: deployment inputs are a different population and must
        # not move the territory the training loop is controlling (two writers, one tau)
        #
        # THE RELIABILITY DEDUCTION. Deployment-only, by construction: there is
        # no y here, so the only thing that can say how far to trust either side
        # is the reliability training MEASURED per centroid (Aman, 2026-09-17:
        # the reliability score is a deduction factor for a specific composition
        # of centroids, for when there is no real data to measure the output
        # against). Training never applies it — it has y, gradients flow at full
        # strength, and standing does the rest.
        #
        # THREE PASSES (Aman, 2026-09-20). Central answers alone; the centroids
        # produce a synthesis out of the individual expert outputs; Central then
        # builds one answer out of both. The split between them is not a
        # constant: each side is scored, and "central's output is leaned towards
        # more and has heavier weight in synthesis ... when central has better
        # score".
        notes.sort(key=lambda x: -x[0])
        if not notes:
            return {"text": self.central.generate(prompt, [], max_tokens), "k": 0,
                    "timeline": "B", "trust": trust, "rho_synth": 0.0, "lean": 1.0,
                    "notes": [], "home": plan.home, "w": plan.w.round(3).tolist()}
        own = self.central.generate(prompt, [], max_tokens)          # 1. Central alone
        synth = "\n".join(t.strip() for _, t in notes if t and t.strip())   # 2. the centroids'
        # 3. the similarity score and the dot products are measured ON THAT
        # SYNTHESIS, not on the input: the synthesis is its own point in the
        # space and lands in its own composition, which is what its reliability
        # has to be read off. The input's composition scores Central; the
        # synthesis's composition scores the synthesis.
        rho_synth, w_synth = 0.0, None
        if synth:
            ids = self.gate.encode(synth)[: C.WORKING_PROBE_TOKENS]
            if ids:
                w_synth = self.geo.compose(self.gate.hidden(ids))[0]
                rho_synth = rho_of(self.rel, w_synth)
        tot = trust + rho_synth
        lean = float(trust / tot) if tot > 1e-12 else 0.5      # Central's share of the merge
        # WHICH OUTPUT IS IN CHARGE is the comparison itself (Aman, 2026-09-20:
        # "the comparison is done against which is more reliability score — if
        # pool then synthesised output (made from expert parts, not final which
        # is answer), if central you know it"). The winner leads the merge and
        # the loser becomes material for it; the deduction, unchanged in shape,
        # still says how much of the pool gets in when Central is the one
        # leading. Central emits the final answer either way — the synthesis is
        # made of expert parts and is never itself the answer.
        if trust >= rho_synth:
            lead, label = "central", "Your own draft answer"
            base = own
            use = [t for _, t in notes[: max(1, int(math.ceil(len(notes) * (1.0 - lean))))]]
        else:
            lead, label = "pool", "The experts' synthesis, which scored higher than your own read"
            base = synth
            use = [own]
        return {"text": self.central.generate(prompt, use, max_tokens, base=base, base_label=label),
                "k": len(notes), "timeline": "B", "trust": trust, "rho_synth": rho_synth,
                "lean": lean, "lead": lead, "notes": use, "own": own, "synth": synth,
                "home": plan.home, "w": plan.w.round(3).tolist(),
                "w_synth": (w_synth.round(3).tolist() if w_synth is not None else None)}
