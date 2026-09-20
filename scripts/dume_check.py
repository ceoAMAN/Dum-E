"""Model-free checks for the dume package. Runs in seconds, no models.

    python scripts/dume_check.py

Each check asserts a property the design DEPENDS on, not a code path:
directions frozen, volume cannot buy standing, migration displaces at the cap
and only on positive evidence, chains never freeze, high traffic tightens,
cold reliability is full trust, tau has a consumer, the trial span really is
small, spans are contiguous and cover the input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import math

from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dume import config as C                                   # noqa: E402
from dume.chain import MarkovChain, MigrationChains, SizeChains  # noqa: E402
from dume.geometry import Geometry                             # noqa: E402
from dume.health import Health                                 # noqa: E402
from dume.reward import CentralBand, Reliability, _split, is_heldout, rho_of   # noqa: E402
from dume.alloc import AllocLaw, probe_sizes
from dume.curriculum import SPECIALIZE, TEST, Curriculum       # noqa: E402
from dume.router import Router                                 # noqa: E402
from dume.standing import GENERAL, MIN_MOVE_OBS, Standing   # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(3, 64))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    X = np.concatenate([d + 0.04 * rng.normal(size=(60, 64)) for d in dirs])

    # geometry
    g = Geometry.form(X, 10, {"gate": "test"})
    assert g.C == 3, f"expected 3 clusters, got {g.C}"
    w, assign, sims, msims = g.compose(X[:60])
    assert np.argmax(w) == np.argmax(np.bincount(assign))
    B0 = g.B.copy()
    g.set_tau(0, 0.5)
    assert np.array_equal(B0, g.B), "direction moved on set_tau"
    # tau has a consumer: tightening removes the cluster from an edge input
    # an EDGE input: mostly cluster 0, partly cluster 1, so its similarity to home
    # sits inside the band tau can actually reach (set_tau clamps at TAU_MAX by design)
    edge = np.concatenate([X[:45], X[60:75]])
    we, _, _, mse = g.compose(edge)
    home = g.home(we)
    assert float(mse[home]) < C.TAU_MAX, f"edge input too central to test tau ({mse[home]:.3f})"
    g.set_tau(home, 0.5)
    assert home in g.inside(mse)
    g.set_tau(home, float(mse[home]) + 1e-3)        # just tighter than this input's similarity
    assert home not in g.inside(mse), "tightening did not remove the cluster from its territory"
    g.set_tau(home, 0.5)
    assert home in g.inside(mse)
    # a singleton direction must not survive formation (no tau from one point)
    Xo = np.vstack([X, (dirs[0] + dirs[1] + dirs[2])[None, :] * 5.0])
    assert Geometry.form(Xo, 10, {"gate": "test"}).C == 3, "outlier became its own cluster"
    print("geometry      OK  (3 clusters, direction frozen, tau gates territory, outlier dropped)")

    # standing
    # cold start: performance has concentrated nowhere, so EVERY expert is
    # general — a candidate anywhere, at home nowhere.
    s = Standing(g.C)
    assert len(s.generals()) == C.E and not any(s.members(c) for c in range(g.C))
    assert s.trial(0, set()) is not None, "no dormant trial available at cold start"
    # standing is a PURE RATE — no confidence, no discount. Confidence in this
    # system exists only at the Timeline A/B gate.
    s.observe(10, 0, 0.5, 32)
    s.observe(11, 0, 0.5, 320)
    assert abs(s.score(10, 0) - 0.5 / 32) < 1e-12, "standing is not sum_delta/sum_tokens"
    assert s.score(10, 0) > s.score(11, 0), "volume bought standing"
    # MEMBERSHIP IS RANK-AND-FILL (Aman, 2026-09-10). No sign test, no 0.90 band:
    # rank by overall rate, top GENERAL_EXPERTS stay general, the rest fill the
    # centroid their own row favours. NEGATIVE rates still place — that is the
    # whole point. The old absolute band deadlocked: 135 batches produced twelve
    # pairs all in [-0.005, -0.000], every position() came back the zero vector,
    # nobody was ever seated and the curriculum could never leave TEST.
    N = C.GENERAL_EXPERTS + 4
    s1 = Standing(2)
    for e in range(N):                       # every rate NEGATIVE
        for _ in range(MIN_MOVE_OBS):
            s1.observe(e, e % 2, -1.0 - e, 32)
    assert s1.migrate(MigrationChains(2)), "all-negative standing seated nobody"
    seated = [e for e in range(N) if s1.assigned[e] != GENERAL]
    assert len(seated) == 4, seated                       # N - GENERAL_EXPERTS
    assert seated == [10, 11, 12, 13], seated             # the BEST ten stay general
    assert all(s1.assigned[e] == e % 2 for e in seated), "placed off its own best row"
    # one observation is not evidence for a seat, however good it looks
    s1.observe(80, 0, 99.0, 32)
    s1.migrate(MigrationChains(2))
    assert s1.assigned[80] == GENERAL, "one batch bought a seat"
    # a seat follows the ROW: e10 sits in c0 on -0.34/token; give it a
    # less-bad c1 row (still negative, so it stays out of the elite) and its
    # home must move. This is the case the clipped position() could not express.
    for _ in range(MIN_MOVE_OBS):
        s1.observe(10, 1, -0.1, 32)
    s1.migrate(mig := MigrationChains(2))
    assert s1.assigned[10] == 1, f"the home did not follow the row: {s1.assigned[10]}"
    assert sum(mig.pool.evidence(i) for i in range(mig.pool.n_states)) > 0, \
        "migration chain did not observe the move"
    # RANKING = normalised weighted sum of correctness, hallucination, throughput
    # (p4). All three are measured: the first two are the same per-token gradient
    # split at zero, the third is wall time.
    s3 = Standing(3)
    s3.observe(20, 0, 1.0, 32, correct=1.0, halluc=0.0, seconds=1.0)
    s3.observe(21, 2, 1.0, 32, correct=1.0, halluc=0.0, seconds=1.0)
    rk = s3.rank(np.array([0.8, 0.1, 0.1]))
    assert rk[20] > rk[21], "domain rank did not weight the overall ranking"
    place = AllocLaw.placement(rk)
    assert place[20] == 1.0 and place[21] == 0.0, place
    # two experts with the SAME net delta but different volatility must not rank
    # equal: correct-minus-halluc is what the old scalar saw, and it is blind here
    s8 = Standing(1)
    s8.observe(1, 0, 0.0, 32, correct=0.0, halluc=0.0, seconds=1.0)   # inert
    s8.observe(2, 0, 0.0, 32, correct=3.0, halluc=3.0, seconds=1.0)   # volatile
    assert s8.score(1, 0) == s8.score(2, 0), "the net rate should be identical"
    r8 = s8.rank(np.array([1.0]))
    # at W_HALLUC == 1 the two gradient terms CANCEL and volatility is invisible.
    # This is not a bug to paper over — it is the algebra, and it is why W_HALLUC
    # is the one coefficient that has to come from Aman.
    assert r8[1] == r8[2], f"equal weights should tie {r8}"
    import dume.standing as _st
    _st.W_HALLUC = 2.0
    try:
        r8b = s8.rank(np.array([1.0]))
        assert r8b[1] > r8b[2], f"W_HALLUC>1 did not make volatility cost {r8b}"
    finally:
        _st.W_HALLUC = 1.0
    # throughput enters: same gradient, half the time -> better rank
    s9 = Standing(1)
    s9.observe(1, 0, 1.0, 32, correct=1.0, halluc=0.0, seconds=2.0)
    s9.observe(2, 0, 1.0, 32, correct=1.0, halluc=0.0, seconds=1.0)
    r9 = s9.rank(np.array([1.0]))
    assert r9[2] > r9[1], f"processing time was ignored {r9}"
    # the general class is ELECTED ONCE and LOCKED: kept and trained, never
    # re-compared. A newcomer that outperforms every general does NOT take the
    # class from one — it takes a SEAT. Re-ranking generals every migrate was
    # churn: at the measured noise the top ten settles to 8.8/10 carry-over even
    # when no expert is better than any other, so a stable set proves nothing.
    sL = Standing(2)
    for e in range(N):
        for _ in range(MIN_MOVE_OBS):
            sL.observe(e, e % 2, -1.0 - e, 32)
    assert sL.settle_generals() == list(range(C.GENERAL_EXPERTS)), sL.settle_generals()
    locked = set(sL.elite)
    for _ in range(MIN_MOVE_OBS):
        sL.observe(13, 0, 500.0, 32)              # a seated expert becomes the best alive
    sL.migrate(MigrationChains(2))
    assert set(sL.elite) == locked, "the general class was re-elected"
    assert sL.assigned[13] == 0, "the new best expert was not placed on its row"
    assert all(sL.assigned[e] == GENERAL for e in locked), "a locked general lost its class"
    # unplaced experts are general by RESIDUE, not election: still ranked, still movable
    assert 13 not in sL.elite and sL.assigned[13] != GENERAL
    # anti-dominance: more claimants than seats -> the weakest go GENERAL, they
    # are NOT shoved into a centroid they do not point at
    s5 = Standing(2)
    for e in range(C.GENERAL_EXPERTS + C.CENTROID_EXPERTS + 3):
        for _ in range(MIN_MOVE_OBS):
            s5.observe(e, 0, 1.0 + e, 32)
    s5.migrate(MigrationChains(2))
    assert len(s5.members(0)) == C.CENTROID_EXPERTS, f"cap violated {s5.members(0)}"
    assert not s5.members(1), "overflow was displaced into a centroid it never earned"
    assert all(s5.assigned[e] == GENERAL for e in range(3)), "the weakest claimants kept seats"
    print("standing      OK  (rank-and-fill; negatives place; general class elected once then LOCKED)")

    # growth: a centroid EARNS seats from the input it receives, ceilinged
    gg = Geometry(np.eye(10, 32).astype(np.float32), np.full(10, 0.5, np.float32), {"gate": "t"})
    assert gg.capacity().tolist() == [0] * 10, "a cold centroid already had seats"
    assert gg.band() == (3, 4), gg.band()                      # C=10: predecessor/successor, not +-1
    assert Geometry(np.eye(9, 32).astype(np.float32), np.full(9, 0.5, np.float32), {}).band() == (2, 4), \
        "perfect square must widen to [sqrt-1, sqrt+1]"
    for n, want in ((1, 1), (4, 2), (81, 9), (400, 9)):        # floor(sqrt), ceilinged at 9
        gn = Geometry(np.eye(2, 32).astype(np.float32), np.full(2, 0.5, np.float32), {})
        gn.grow([0] * n)
        assert int(gn.capacity()[0]) == want, (n, gn.capacity()[0], want)
    # a centroid with 1 earned seat seats exactly its best claimant; the rest general
    s6 = Standing(2)
    for e in range(C.GENERAL_EXPERTS + 4):
        for _ in range(MIN_MOVE_OBS):
            s6.observe(e, 0, 1.0 + e, 32)
    g6 = Geometry(np.eye(2, 32).astype(np.float32), np.full(2, 0.5, np.float32), {})
    g6.grow([0])                                               # one input -> one seat
    s6.migrate(MigrationChains(2), cap=g6.capacity())
    assert s6.members(0) == [3], s6.members(0)
    assert len(s6.generals()) == C.E - 1, "growth did not bound the class"
    # curriculum: TRAINING sweeps the pool, it does not route. The router
    # concentrated 44 of 50 observations on four experts and starved 66.
    cur = Curriculum(n_experts=12, seed=1)
    s7 = Standing(2)
    counts = {}
    for _ in range(3 * (12 // 4)):                 # three full sweeps at k=4
        for e in cur.experts(4, 0, s7):
            counts[e] = counts.get(e, 0) + 1
    assert len(counts) == 12, f"a sweep missed experts: {sorted(counts)}"
    assert set(counts.values()) == {3}, f"unequal exposure {sorted(counts.values())}"
    assert cur.phase == TEST and cur.sweeps >= 2, cur.state()
    # a group is CONSISTENT: the same call does not reshuffle mid-batch
    assert len(set(cur.experts(4, 0, s7))) == 4, "a batch repeated an expert"
    # not every expert is rankable yet -> testing is not over
    assert not cur.ready(s7) and not cur.advance(s7), "promoted before the pool was measured"
    for e in range(s7.E):
        for _ in range(MIN_MOVE_OBS):
            s7.observe(e, 1 if e % 3 else 0, 1.0 + e, 32)
    assert cur.ready(s7), "full coverage did not end testing"
    s7.settle_generals()
    s7.migrate(MigrationChains(2))
    assert cur.advance(s7) and cur.phase == SPECIALIZE, (cur.state(), s7.members(1))
    # a membership change must NOT restart the rota: migration thrashes at this
    # SNR, so restarting starves whoever sorts late in members().
    cur2 = Curriculum(n_experts=40, seed=3); cur2.phase = SPECIALIZE
    s10 = Standing(2)
    for e in (10, 11, 12, 13):
        for _ in range(MIN_MOVE_OBS):
            s10.observe(e, 0, 1.0, 32)
    s10.assigned[[10, 11, 12, 13]] = 0
    seen10 = {}
    for step in range(12):
        for e in cur2.experts(2, 0, s10):
            seen10[e] = seen10.get(e, 0) + 1
        if step == 5:                      # membership changes mid-stream
            s10.observe(14, 0, 1.0, 32); s10.observe(14, 0, 1.0, 32)
            s10.assigned[14] = 0
    early = sum(seen10.get(e, 0) for e in (10, 11))
    late = sum(seen10.get(e, 0) for e in (12, 13))
    assert late >= early - 2, f"rota restart starved the late members {seen10}"
    # specialize: the routed centroid is served by ITS OWN members, cycled
    got = set()
    for _ in range(4):
        got |= set(cur.experts(2, 1, s7))
    assert got and got <= set(s7.members(1)), (got, s7.members(1))
    print(f"curriculum    OK  (sweep gives all 12 equal exposure; test -> specialize on "
          f"{len(s7.members(1))} seated; centroid rota)")

    print("growth        OK  (seats = floor(sqrt(seen)) <= 9; cold = 0; sqrt(C) band 10->[3,4], 9->[2,4])")

    # chains
    mc = MarkovChain(3, memory=10)
    for _ in range(1000):
        mc.observe(0, 1)
    mc.observe(0, 2)
    pb = mc.predict(0)[2]
    for _ in range(5):
        mc.observe(0, 2)
    assert mc.predict(0)[2] > pb, "chain froze (1/n)"
    # territory: a cluster present far more often than average must TIGHTEN, a
    # never-present one must widen, and the average cluster must hold still
    sz = SizeChains(g.C)
    for _ in range(60):
        sz.observe([0, 1])          # 0 and 1 always inside; 2 never
    assert sz.next_tau(0, 0.7) > 0.7, "over-present cluster must tighten"
    assert sz.next_tau(2, 0.7) < 0.7, "never-present cluster must widen"
    sz2 = SizeChains(3)
    for _ in range(60):
        sz2.observe([0, 1, 2])      # everyone equally present == the setpoint
    assert sz2.next_tau(0, 0.7) == 0.7, f"balanced load must not move tau (got {sz2.next_tau(0, 0.7)})"
    print("chains        OK  (bounded evidence; tighten on presence; balanced load is a fixed point)")

    # reliability
    r = Reliability(g.C)
    assert r.R(0) == 1.0, "cold must be full trust"
    r.observe(np.full(200, 0.5), np.zeros(200, dtype=int))
    r.observe(np.full(200, 3.0), np.ones(200, dtype=int))
    assert r.R(0) > r.R(1)
    # R TRACKS CURRENT PERFORMANCE. Reliability is applied at deployment but it
    # moves in training, because that is where the model moves. A lifetime mean
    # could not: it stiffens as 1/N. After a long bad history, a window of good
    # observations must win.
    rt = Reliability(2)
    rt.observe(np.full(20000, 3.0), np.zeros(20000, dtype=int))
    stale = rt.R(0)
    W = C.RELIABILITY_MIN_OBS
    rt.observe(np.full(4 * W, 0.1), np.zeros(4 * W, dtype=int))
    assert rt.R(0) > 0.80, f"four windows of good data did not overcome 20k bad ({rt.R(0)})"
    rt.observe(np.full(4 * W, 0.1), np.zeros(4 * W, dtype=int))
    assert abs(rt.R(0) - np.exp(-0.1)) < 0.01, (stale, rt.R(0), np.exp(-0.1))
    # ... and 20k observations of history do not hold it back
    assert rt.N[0] == 20000 + 8 * W, "support counter must not decay"
    # states pickled with a cumulative S migrate to the EWMA at the SAME R
    old = Reliability(2)
    old.__dict__.clear()
    old.__setstate__({"N": np.array([400.0, 100.0]), "S": np.array([200.0, 50.0])})
    assert not hasattr(old, "S") and abs(old.R(0) - np.exp(-0.5)) < 1e-9, "S -> M migration changed R"

    # rho is the COMPOSITION's score: percentage in the constituent x that
    # centroid's measured mean. Not a per-token mean over the target, and not
    # per-cluster: a centroid absent from the composition has w=0 and cannot
    # move it, however good or bad its own row is.
    w = np.zeros(g.C); w[0], w[1] = 0.75, 0.25
    # 1e-7, not 1e-12: vector() is float32 and rho_of reads R through it.
    assert abs(rho_of(r, w) - (0.75 * r.R(0) + 0.25 * r.R(1))) < 1e-7, "rho is not the weighted mean"
    w_all0 = np.zeros(g.C); w_all0[0] = 1.0
    assert abs(rho_of(r, w_all0) - r.R(0)) < 1e-7, "a pure composition must be that centroid's own R"
    assert rho_of(r, w) < rho_of(r, w_all0), "the bad centroid's share must pull rho down"
    try:
        rho_of(r, np.zeros(g.C + 3))
        raise AssertionError("rho accepted a composition of the wrong width")
    except RuntimeError:
        pass
    # R is MEASURED in training, never applied to the delta: the reward is a plain
    # mean over tokens. A token on an unreliable centroid counts exactly as much as
    # one on a reliable centroid — the delta already carries the headroom.
    d = np.array([1.0, 1.0, -0.5, -0.5])
    net, cor, hal = _split(d)
    assert abs(net - d.mean()) < 1e-12 and abs(cor - 0.5) < 1e-12 and abs(hal - 0.25) < 1e-12, (net, cor, hal)
    frac = sum(is_heldout(f"k{i}") for i in range(4000)) / 4000
    assert 0.2 < frac < 0.3, frac
    # THE RELIABILITY DEDUCTION IS DEPLOYMENT-ONLY. Training has y, so gradients
    # flow at full strength; rho is MEASURED there and APPLIED in answer(), where
    # ceil(n_notes * (1 - trust)) decides how much of the synthesis the experts
    # carry. Pins the deployment formula, and that nothing in the training path
    # scales by rho any more.
    import inspect as _insp
    from dume import train as _tr, models as _md
    assert "sc.rho" not in _insp.getsource(_tr.System._expert_update), "expert advantage still rho-scaled"
    assert "rho" not in _insp.signature(_md.ExpertPool.sample).parameters, "sample() still takes rho"
    assert "weight" not in _insp.signature(_md.Central.pretrain_step).parameters, "Central step still weighted"
    src_ans = _insp.getsource(_tr.System.answer)
    assert "math.ceil(len(notes) * (1.0 - lean))" in src_ans, "deployment deduction formula changed"
    # THREE PASSES, and the split between them is a COMPARISON, not a constant:
    # Central alone, the centroids' synthesis, then one answer built from both.
    # The dot products that score the synthesis are measured on the synthesis.
    assert "self.central.generate(prompt, [], max_tokens)" in src_ans, "Central no longer drafts alone"
    assert 'lead, label = "central"' in src_ans and 'lead, label = "pool"' in src_ans, \
        "the comparison no longer decides which output is in charge"
    assert "base=base, base_label=label" in src_ans, "the merge base is not the winner of the comparison"
    assert "self.geo.compose(self.gate.hidden(ids))" in src_ans, "synthesis is not composed"
    assert "trust / tot" in src_ans, "the lean is not a comparison of the two scores"
    assert "base" in _insp.signature(_md.Central.generate).parameters, "generate() has no base slot"
    # the training path scores the COMPOSITION, never the target's assignment
    src_one_w = _insp.getsource(_tr.System._train_one)
    assert "score(self.central, s.prompt, y, plan.w," in src_one_w, "training rho is not composition-weighted"
    assert "self.rel.observe(sc.b, assign_y)" in src_one_w, "per-token measurement of R was lost"
    # Central's step: outside admission, and guarded against held-out AND self-referent
    src_one = _insp.getsource(_tr.System._train_one)
    i_adm, i_cen = src_one.index("elif sc.admitted:"), src_one.index("self.central.pretrain_step(")
    assert i_cen > i_adm, "Central step precedes the admitted block"
    guard = src_one[src_one.rindex("if not heldout", 0, i_cen):i_cen]
    assert "not s.self_referent" in guard, "Central can train on its own delivered answer"
    assert "self.central.save()" in _insp.getsource(_tr.System.save), "Central's in-loop training is not persisted"
    print(f"reliability   OK  (cold=1.0; vector {r.vector().round(2)}; misalignment refused; heldout {frac:.2f})")

    # allocation law: probe schedule, per-expert spread, admissibility, log clearing
    law = AllocLaw(100)
    assert probe_sizes(53) == (4, 26, 53) and probe_sizes(50)[0] == 1, probe_sizes(53)
    def _d(t):   return math.log1p(t / 40.0) - t / 600.0
    def _lat(t): return 0.40 + 0.004 * t
    rng2 = np.random.default_rng(0)
    for T in [int(x) for x in np.exp(rng2.uniform(math.log(24), math.log(1400), 300))]:
        for t in probe_sizes(T):
            law.observe(t, _d(t) + rng2.normal(0, 0.05), _lat(t))
        law.input_seen(T)
        if law.n_seen % law.E == 0:
            law.fit()
    assert law.fitted and 0 < law.beta <= 1.0, law.state()
    assert not law.q and not law.lat and not law.peaks, "logs survived a refit"
    assert law.cost[0] > 0.0, "cost intercept collapsed to zero — argmax pins at t_lo"
    ks = [law.k(T) for T in (32, 128, 512, 1354)]
    assert all(b >= a for a, b in zip(ks, ks[1:])), f"k decreased with T: {ks}"
    assert all(law.k(T) >= 1 for T in (1, 2, 1354)), "k gated an expert to zero"
    # the allocation is a SOFT limit: floor 32, ceilinged by the span read and by
    # the expert's measured share of Central's context. No exponential blowup.
    assert AllocLaw.budget(900, 1354, 31616) == C.TARGET_MAX_TOKENS, "y-length ceiling did not bind"
    assert AllocLaw.budget(900, 40, 31616) == 40, "span did not bind"
    assert AllocLaw.budget(5, 1354, 31616) == C.EXPERT_GEN_TOKENS, "floor did not bind"
    assert AllocLaw.budget(4, 8, 31616) == C.EXPERT_GEN_TOKENS, "floor lost to a tiny span"
    # THE ceiling that matters: generation can never run away, whatever is fitted.
    # Central's nominal 32k context share does NOT clamp — relying on it crashed
    # the machine by asking one expert to generate a whole input's worth of text.
    assert max(AllocLaw.budget(a, sp, sh) for a in (1, 50, 10**6)
               for sp in (8, 300, 10**5) for sh in (32, 700, 31616)) <= C.TARGET_MAX_TOKENS, \
        "generation budget escaped TARGET_MAX_TOKENS"
    narrow = AllocLaw(100)
    narrow.peaks = [(40.0, 12.0), (42.0, 13.0), (44.0, 14.0)]
    narrow._fit_law()
    assert not narrow.fitted and narrow.alloc(500) == 500 and narrow.k(500) == 1, "narrow span was fitted"
    convex = AllocLaw(100)
    for t, q in [(10, 0.0), (50, 0.0), (300, 5.0)] * 4:      # curving UP: no interior peak
        convex.observe(t, q, 1.0)
    convex._fit_envelopes()
    assert convex.A is None and "convex" in convex.last_reject, convex.last_reject
    print(f"alloc         OK  (probe from T alone; beta {law.beta:.3f}; k non-decreasing; "
          f"convex + narrow-span refused; logs cleared)")

    # router spans: anchors ordered and contiguous, every span padded to its allocation
    T = 400
    assign_t = np.array([0] * 200 + [1] * 200)
    picks = [(50, 0, False), (60, 1, False), (70, 0, True)]
    rt = Router.__new__(Router)
    rt.alloc = law
    rt.sched = SimpleNamespace(span_max=1 << 30)          # unbounded for the geometry checks
    spans = rt._spans(picks, assign_t, T, False, {50: 1.0, 60: 0.5, 70: 0.0})
    assert spans[0].start == 0, [(s.start, s.end) for s in spans]
    for a, b in zip(spans, spans[1:]):
        assert b.start >= a.start, "anchors out of order"
    assert all(0 <= s.start < s.end <= T for s in spans), "span left the input"
    probe_spans = rt._spans(picks, assign_t, T, True, {})
    assert {s.n_tokens for s in probe_spans} == set(probe_sizes(T)), \
        f"probe spans {[s.n_tokens for s in probe_spans]} != {probe_sizes(T)}"
    # at k=1 the probe must still sweep all three sizes, across BATCHES
    seen1 = set()
    for b in range(3):
        law.n_seen = b
        seen1.add(rt._spans([picks[0]], assign_t, T, True, {})[0].n_tokens)
    assert seen1 == set(probe_sizes(T)), f"k=1 probe stuck at {seen1}"
    # the MEMORY bound on span: the backward pass is linear in prompt length, so
    # t_hi may not be T on a long row. Same physical clamp as k_max on k.
    rt.sched = SimpleNamespace(span_max=64)
    bounded = rt._spans(picks, assign_t, T, True, {})
    assert max(s.n_tokens for s in bounded) <= 64, [s.n_tokens for s in bounded]
    assert set(s.n_tokens for s in bounded) == set(probe_sizes(64)), \
        "the probe must still sweep three sizes inside the bound, not collapse to it"
    rt.sched = SimpleNamespace(span_max=1 << 30)
    print(f"router        OK  (anchors ordered, spans within input; probe spans "
          f"{sorted(s.n_tokens for s in probe_spans)}; span_max clamps t_hi)")

    # Timeline A/B: decided by the MEASURED gain, not a quantile and not a constant
    band = CentralBand()
    cold = AllocLaw(100)
    assert band.decide(cold, 500, 0.9) == "B", "Timeline A fired with no curves fitted"
    rates = []
    for level in (1.2, 0.2, -0.6):
        lw, bd = AllocLaw(100), CentralBand()
        r4 = np.random.default_rng(1)
        for _ in range(400):
            Tq = int(np.clip(np.exp(r4.normal(math.log(180), 1.3)), 10, 4000))
            for t in probe_sizes(Tq):
                lw.observe(t, level + 0.3 * math.log1p(t / 300.0) + r4.normal(0, 0.2), 0.4 + 0.004 * t)
            lw.input_seen(Tq)
            if lw.n_seen % 100 == 0:
                lw.fit()
        for _ in range(300):
            bd.decide(lw, int(np.clip(np.exp(r4.normal(math.log(180), 1.3)), 10, 4000)), r4.random())
        rates.append(bd.rate())
    assert rates[0] == 0.0, "Timeline A fired while experts were clearly helping"
    assert rates[-1] > 0.9, f"Timeline A did not take over once the gain went negative: {rates}"
    assert rates[0] <= rates[1] <= rates[2], f"Timeline A not monotone in gain: {rates}"
    print(f"alloc/band    OK  (no curves -> all B; A rate {[round(r,2) for r in rates]} "
          f"as measured gain crosses zero)")

    # health
    h = Health()
    h.put(x=float("nan"))
    assert h.nonfinite == 1
    h.update_seen(None); h.update_seen(1.0)
    assert h.updates == 1 and list(h.applied) == [0.0, 1.0], "skipped update counted as applied"
    # the update floor must be ROBUST: one short-answer outlier inflated
    # arr.std() enough to refuse 96% of gradient steps over 1700 batches.
    from dume.health import mad_sigma
    clean = np.random.default_rng(0).normal(0, 0.05, 50)
    dirty = np.concatenate([clean, [2.90]])
    assert abs(mad_sigma(clean) - clean.std()) < 0.02, "MAD disagrees with std on clean data"
    assert mad_sigma(dirty) < 2 * mad_sigma(clean), "one outlier moved the robust spread"
    assert dirty.std() > 4 * clean.std(), "the std should be the fragile one"
    print("health        OK  (non-finite caught at record time; skipped update is not a loss)")
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
