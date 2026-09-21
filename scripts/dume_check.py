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
    # THE GENERAL CLASS IS A SPACE, NOT A SET OF MEMBERS (Aman, 2026-09-20:
    # "generals aren't unreachable elites ... if someone performs better it can
    # replace it"). e10 is seated in c0; give it a far better c1 row and it must
    # take a general seat from an incumbent, who drops back to a centroid.
    before = sorted(s1.elite)
    for _ in range(MIN_MOVE_OBS):
        s1.observe(10, 1, -0.1, 32)
    s1.migrate(mig := MigrationChains(2))
    assert len(s1.elite) == C.GENERAL_EXPERTS, f"the space changed size: {len(s1.elite)}"
    assert 10 in s1.elite, f"a better performer did not replace an incumbent: {sorted(s1.elite)}"
    displaced = set(before) - set(s1.elite)
    assert len(displaced) == 1, f"exactly one incumbent must fall out, got {displaced}"
    assert s1.assigned[list(displaced)[0]] != GENERAL, "the displaced general took no centroid seat"
    # a seat still follows the ROW for an expert that does NOT reach the class:
    # e13 is the worst of the pool; its best row must decide its home.
    for _ in range(MIN_MOVE_OBS):
        s1.observe(13, 0, -13.0, 32)
    s1.migrate(MigrationChains(2))
    assert 13 not in s1.elite and s1.assigned[13] == 0, \
        f"the home did not follow the row: elite={13 in s1.elite} assigned={s1.assigned[13]}"
    assert sum(mig.pool.evidence(i) for i in range(mig.pool.n_states)) > 0, \
        "migration chain did not observe the move"
    # RANKING = normalised weighted sum of correctness, hallucination, throughput
    # (p4). All three are measured: the first two are the same per-token gradient
    # split at zero, the third is wall time.
    # DOMAIN WEIGHTING: both experts are measured in the SAME two clusters, so
    # the only thing separating them is WHERE they are good. Being good in the
    # 0.8 domain must beat being equally good in the 0.1 domain. (The old test
    # gave each expert one cluster and relied on the unmeasured cells scoring
    # 0.0 — which only ranks correctly when rates are positive, and 202 of the
    # 222 measured cells in the live state are negative, so that dot product
    # preferred the specialist in the SMALL domain exactly when it mattered.)
    s3 = Standing(3)
    for _ in range(MIN_MOVE_OBS):
        s3.observe(20, 0, 1.0, 32, correct=2.0, halluc=0.0, seconds=1.0)   # good where it counts
        s3.observe(20, 2, 1.0, 32, correct=1.0, halluc=0.0, seconds=1.0)
        s3.observe(21, 0, 1.0, 32, correct=1.0, halluc=0.0, seconds=1.0)
        s3.observe(21, 2, 1.0, 32, correct=2.0, halluc=0.0, seconds=1.0)   # good in the small one
    rk = s3.rank(np.array([0.8, 0.1, 0.1]))
    assert rk[20] > rk[21], f"domain rank did not weight the overall ranking {rk}"
    # and an expert measured NOWHERE ELSE must not beat one measured everywhere
    # just by leaving cells blank (the ignorance bias, on negative rates)
    s3b = Standing(3)
    for _ in range(MIN_MOVE_OBS):
        s3b.observe(30, 0, -0.5, 32)                       # thin, one cluster
        for c in range(3):
            s3b.observe(31, c, -0.5, 32)                   # same rate, measured everywhere
    ov = s3b._overall(s3b.rates(), np.array([0.8, 0.1, 0.1]), s3b.n >= MIN_MOVE_OBS)
    assert ov[30] <= ov[31] + 1e-12, f"thin evidence outscored full evidence: {ov[30]} vs {ov[31]}"
    place = AllocLaw.placement(rk)
    assert place[20] == 1.0 and place[21] == 0.0, place
    # THREE SEPARATE TERMS, each on its own normaliser: efficiency, NON-
    # hallucination, throughput. They used to share one scale and enter as
    # (cor - hal), so the hallucination weight was inert by construction.
    #
    # AND THE WEIGHT IS MEASURED, not configured: w_halluc() = total nats
    # hallucinated / total nats saved, both accumulated against real y.
    s8 = Standing(1)
    s8.observe(1, 0, 0.0, 32, correct=2.0, halluc=1.0, seconds=1.0)
    s8.observe(2, 0, 0.0, 32, correct=4.0, halluc=3.0, seconds=1.0)   # same net, 3x the halluc
    assert s8.score(1, 0) == s8.score(2, 0), "the net rate should be identical"
    # this pool hallucinated 4 nats against 6 saved -> weight 2/3, below the
    # neutral pair, so a pool that mostly helps does NOT get a hallucination-led
    # ranking. The number comes off the books, nobody chose it.
    assert abs(s8.w_halluc() - (4.0 / 6.0)) < 1e-12, s8.w_halluc()
    # at that weight e2's larger correctness outweighs its larger hallucination,
    # and it SHOULD: the pool is net-helpful, so the books say correctness is
    # the scarcer thing. The weight follows the data, not a preference.
    r8 = s8.rank(np.array([1.0]))
    assert r8[2] > r8[1], f"a net-helpful pool should still reward correctness {r8}"
    # a pool that mostly HURTS measures a weight above 1 and leans harder on it
    s8b = Standing(1)
    s8b.observe(1, 0, 0.0, 32, correct=1.0, halluc=2.0, seconds=1.0)
    s8b.observe(2, 0, 0.0, 32, correct=1.0, halluc=8.0, seconds=1.0)
    assert s8b.w_halluc() == 10.0 / 2.0, s8b.w_halluc()
    # THE INVARIANT, at any weight: hold correctness equal and more
    # hallucination must rank lower. This is what "rewards non hallucination"
    # means, and the old (cor - hal) form could not express it.
    r8b = s8b.rank(np.array([1.0]))
    assert r8b[1] > r8b[2], f"equal correctness, more hallucination must lose {r8b}"
    # a cold pool has nothing to measure and falls back to the neutral pair
    assert Standing(1).w_halluc() == 1.0, "cold pool must not invent a weight"
    # throughput enters: same gradient, half the time -> better rank
    s9 = Standing(1)
    s9.observe(1, 0, 1.0, 32, correct=1.0, halluc=0.0, seconds=2.0)
    s9.observe(2, 0, 1.0, 32, correct=1.0, halluc=0.0, seconds=1.0)
    r9 = s9.rank(np.array([1.0]))
    assert r9[2] > r9[1], f"processing time was ignored {r9}"
    # the general class is a SPACE OF FIXED SIZE, not a fixed set of members
    # (Aman, 2026-09-20). The size never changes; a newcomer that outperforms an
    # incumbent TAKES its place and the incumbent drops to a centroid seat.
    sL = Standing(2)
    for e in range(N):
        for _ in range(MIN_MOVE_OBS):
            sL.observe(e, e % 2, -1.0 - e, 32)
    assert sL.settle_generals() == list(range(C.GENERAL_EXPERTS)), sL.settle_generals()
    was = set(sL.elite)
    for _ in range(MIN_MOVE_OBS):
        sL.observe(13, 0, 500.0, 32)              # a seated expert becomes the best alive
    sL.migrate(MigrationChains(2))
    assert len(sL.elite) == C.GENERAL_EXPERTS, f"the space changed size: {len(sL.elite)}"
    assert 13 in sL.elite, "the best expert alive did not take a general seat"
    out = was - set(sL.elite)
    # >= 1, not exactly 1: an unmeasured cell is imputed from the POOL, so one
    # expert's new result moves everyone's estimate in that cluster. The
    # contract is the SIZE and the replacement, not which incumbent falls.
    assert out, "a better performer displaced nobody"
    assert all(sL.assigned[e] != GENERAL for e in out), "a displaced general took no centroid seat"
    assert sL.assigned[13] == GENERAL, "a general must not also hold a centroid seat"
    # and generals are REACHABLE: the curriculum must be able to seat them, or
    # they can never be measured and never replaced
    from dume.curriculum import Curriculum, SPECIALIZE
    cg = Curriculum(n_experts=N); cg.phase = SPECIALIZE
    drawn = set()
    for _ in range(40):
        drawn |= set(cg.experts(2, 0, sL))
    assert drawn & set(sL.elite), f"the curriculum cannot reach a general: drew {sorted(drawn)}"
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
    print("standing      OK  (rank-and-fill; negatives place; general class is a fixed-size space, membership replaceable)")

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
    # THE DEVICE'S SIDE OF THE TUG OF WAR over k. k_gate pulls up, the device
    # pulls down, and its input is real: NSProcessInfo's thermal state, not a
    # proxy and not a constant.
    from dume.models import thermal_state as _ts, die_temp
    from dume.chain import ThermalRegulator
    assert _ts() in (0, 1, 2, 3), f"thermal state out of range: {_ts()}"
    # THE SENSOR IS REAL. The ordinal is a CONSTANT on this machine -- `fair`
    # on all 580 samples of a 2900-batch run and all 541 of the next -- so the
    # regulator built on it never once moved k. Degrees are what moves.
    t = die_temp()
    assert t is not None, "no die sensors: the regulator is back on a constant"
    assert 0.0 < t < 120.0, f"die temperature implausible: {t}"

    # THE REGULATOR IS NOT REACTIVE. It measures against where this machine
    # NORMALLY sits, over the range it has actually worked in.
    def feed(temps, level=0.0):
        r = ThermalRegulator()
        for x in temps:
            r.observe(x, level)
        return r
    steady = feed([52.0] * 200)                # warm, but warm is its normal
    assert steady.pressure() == 0.0, "a machine that always runs warm was throttled for it"
    assert steady.k_thermal(4.0) == 4.0, "a settled machine did not get the full bound"
    assert ThermalRegulator().pressure() == 0.0, "a cold regulator invented pressure"
    assert ThermalRegulator().k_thermal(4.0) == 4.0, "a cold regulator throttled"

    # THE SYSTEM'S SIDE: small input -> max k and less time; longer input ->
    # fewer experts, because past one pass another expert buys no coverage and
    # costs another fixed load-in.
    SM = 446
    short = [law.k_effective(T, 16, 16.0, SM) for T in (16, 49, 128, 446)]
    long_ = [law.k_effective(T, 16, 16.0, SM) for T in (1024, 2335, 4096)]
    assert all(k == 16 for k in short), f"a small input did not get max k: {short}"
    assert long_ == sorted(long_, reverse=True), f"k did not fall with length: {long_}"
    assert long_[-1] < short[-1], f"a long input did not reduce k: {short[-1]} -> {long_[-1]}"
    # ... and that is the OPPOSITE of what the allocation law's own k(T) asks
    # for, which rises with T. k(T) is still reported as k_wanted; it no longer
    # drives k.
    assert law.k(4096) > law.k(16), "k_wanted should still rise with T"
    # cooling raises k back: the relation runs both ways
    warm = [law.k_effective(128, 16, 16.0 ** (1.0 / (1.0 + st)), SM) for st in (3, 2, 1, 0)]
    assert warm == sorted(warm), f"cooling did not raise k: {warm}"
    kt = [16.0 ** (1.0 / (1.0 + st)) for st in range(4)]      # k_max=16
    ks = [law.k_effective(512, 16, t) for t in kt]
    assert ks == sorted(ks, reverse=True), f"heat did not lower k: {ks}"
    assert ks[0] > ks[-1], f"thermal pressure had no effect at all: {ks}"
    assert all(k >= 1 for k in ks), ks
    # nominal costs nothing: a device asking for nothing leaves the physical
    # bound exactly where it was
    assert law.k_effective(512, 16, kt[0]) == law.k_effective(512, 16), \
        "nominal thermal state moved k"
    # the device can never push k ABOVE what RAM allows — that would be an OOM,
    # not a preference
    assert law.k_effective(512, 4, 999.0) <= 4, "thermal term escaped the RAM clamp"
    # a machine with no sensor is not a hot machine
    assert law.k_effective(512, 16, None) == law.k_effective(512, 16, kt[0]), \
        "a missing sensor quietly throttled the pool"
    print(f"alloc         OK  (probe from T alone; beta {law.beta:.3f}; k non-decreasing; "
          f"convex + narrow-span refused; logs cleared)")

    # router spans: anchors ordered and contiguous, every span padded to its allocation
    T = 400
    assign_t = np.array([0] * 200 + [1] * 200)
    picks = [(50, 0, False), (60, 1, False), (70, 0, True)]
    rt = Router.__new__(Router)
    rt.alloc = law
    rt.sched = SimpleNamespace(span_max=1 << 30)          # unbounded for the geometry checks
    spans = rt._spans(picks, assign_t, T, False)
    assert spans[0].start == 0, [(s.start, s.end) for s in spans]
    for a, b in zip(spans, spans[1:]):
        assert b.start >= a.start, "anchors out of order"
    assert all(0 <= s.start < s.end <= T for s in spans), "span left the input"
    # TRAINING = FAIR SHARE: the allocated tokens divided by the experts sharing
    # them. Equal spans, contiguous anchors, so the k experts TILE the input —
    # everybody is judged on the same amount of material, which is what makes
    # their deltas comparable.
    probe_spans = rt._spans(picks, assign_t, T, True)
    assert len({s.n_tokens for s in probe_spans}) == 1, \
        f"training spans are not an equal share: {[s.n_tokens for s in probe_spans]}"
    assert probe_spans[0].n_tokens == T // len(picks), \
        f"share is not T/k: {probe_spans[0].n_tokens} vs {T // len(picks)}"
    covered = {t for sp in probe_spans for t in range(sp.start, sp.end)}
    assert len(covered) >= T - len(picks), f"the shares did not tile the input: {len(covered)}/{T}"
    # one expert gets the whole allocation, not a third of it
    assert rt._spans([picks[0]], assign_t, T, True)[0].n_tokens == T, "k=1 did not get the whole input"
    # and the share still varies enough across (T, k) for the envelopes to fit
    widths = {rt._spans(picks[:k], assign_t, TT, True)[0].n_tokens
              for TT in (32, 96, 400) for k in (1, 2, 3)}
    assert len(widths) >= 3, f"fair share collapsed the span range: {sorted(widths)}"
    # the MEMORY bound on span: the backward pass is linear in prompt length, so
    # t_hi may not be T on a long row. Same physical clamp as k_max on k.
    rt.sched = SimpleNamespace(span_max=64)
    bounded = rt._spans(picks, assign_t, T, True)
    assert max(s.n_tokens for s in bounded) <= 64, [s.n_tokens for s in bounded]
    # the share is divided out of what the MEMORY bound allows, not out of T
    assert bounded[0].n_tokens == 64 // len(picks), \
        f"the share ignored span_max: {bounded[0].n_tokens} vs {64 // len(picks)}"
    rt.sched = SimpleNamespace(span_max=1 << 30)
    # FRAGMENT CYCLES: the k deficit is paid on the experts already held, so the
    # swap count is per EXPERT and does not grow with coverage (Aman,
    # 2026-09-20: "increase token fragment cycles on them, so then we avoid
    # swapping k experts on regular intervals"). Deployment only.
    rt.sched = SimpleNamespace(span_max=50)   # short spans, so a region needs tiling
    cyc = rt._spans([(3, 0, False), (7, 1, False)], assign_t, T, False, cycles=6)
    order = [x.eid for x in cyc]
    swaps = sum(1 for i in range(len(order)) if i == 0 or order[i] != order[i - 1])
    assert len(cyc) > 2, f"cycles produced no extra fragments: {len(cyc)}"
    assert swaps == 2, f"a fragment cost a swap: {swaps} over {order}"
    assert len(set(order)) == 2, "cycles changed WHICH experts run"
    assert all(0 <= x.start < x.end <= T for x in cyc), "a fragment left the input"
    one = rt._spans([(3, 0, False), (7, 1, False)], assign_t, T, False)
    cov_1 = len({t for x in one for t in range(x.start, x.end)})
    cov_n = len({t for x in cyc for t in range(x.start, x.end)})
    assert cov_n > cov_1, f"cycles bought no coverage: {cov_1} -> {cov_n}"
    # bounded by COVERAGE: a region already tiled buys no duplicate passes
    assert len(rt._spans([(3, 0, False)], assign_t, T, False, cycles=9999)) <= T
    # and the training path is untouched
    assert len(rt._spans([(3, 0, False)], assign_t, T, True, cycles=9)) == 1, \
        "cycles leaked into the probe (training) path"
    rt.sched = SimpleNamespace(span_max=1 << 30)
    # THE EXPERT SEES ITS FRAGMENT PLUS A MAP IT CANNOT MISTAKE FOR MATERIAL.
    # If the input rides along with every span then apex-nadir allocates labels,
    # not budget: the pool costs k*T and dividing the input buys nothing. The
    # orientation is allowed, but it is COMPRESSED (the gate writes it) and it is
    # sized by the law, never by a constant.
    class _Tok:                                   # words as tokens; enough to pin the shape
        def encode(self, s): return s.split()
        def decode(self, t): return " ".join(t)
    _pool = SimpleNamespace(tok=_Tok())
    from dume.models import ExpertPool as _EP
    from dume import train as _tr_mod
    from dume.train import orientation as _orient
    import inspect as _iP
    assert list(_iP.signature(_EP.prompt).parameters) == ["self", "span_text", "context"], \
        "prompt() takes something other than the span and its context"
    span = " ".join(f"s{i}" for i in range(12))
    ctx = " ".join(f"c{i}" for i in range(5))
    built = _EP.prompt(_pool, span, ctx)
    # EXACT ACCOUNTING, not a substring probe. The earlier version asserted that a
    # question's first token was absent from a prompt that had never been given a
    # question, three times over the same dead value — it could not fail. Pin the
    # whole output instead: system + context + span and NOTHING ELSE, so any future
    # smuggled-in field shows up as surplus tokens here.
    surplus = [w for w in built.split() if w not in set(span.split()) | set(ctx.split())]
    boiler = [w for w in _EP.prompt(_pool, "", "").split()]
    assert sorted(surplus) == sorted(boiler), \
        f"the prompt carries {len(surplus) - len(boiler)} tokens that are neither span, context, nor system"
    assert span in built and ctx in built, "the span or its context is missing from the prompt"
    assert built.index(ctx) < built.index(span), "the map does not precede the material it orients"
    # length tracks span and context ONLY, so k experts cost T between them, not k*T
    grew = _EP.prompt(_pool, " ".join(f"s{i}" for i in range(120)), ctx)
    assert len(grew.split()) - len(built.split()) == 108, "prompt carries a length that is not the span"
    # THE POSITION LINE IS THE PART THAT DIFFERS. The old prompt re-printed the
    # excerpt with no indication of where it sat, so the k prompts shared everything
    # that carried meaning. Every expert's orientation must be distinct.
    seen = {_orient("", i, 4, i * 12, (i + 1) * 12, 48) for i in range(4)}
    assert len(seen) == 4, f"orientation did not differentiate the experts: {seen}"
    assert all("of 48" in o for o in seen), "position is not stated in the input's own units"
    withmap = _orient("a map", 0, 4, 0, 12, 48)
    assert withmap.startswith("a map") and "Section 1 of 4" in withmap, "the map and the position do not compose"
    # THE CONTEXT BUDGET IS APEX-NADIR'S SMALLEST ALLOCATION, and below the floor a
    # note has to clear there is no map at all — a two-token summary costs a full
    # prefill to say nothing. _Gate stands in for the real one: summarise() must
    # refuse on the budget BEFORE it reaches the model.
    # the stub RAISES if the guard is passed, so a removed guard fails loudly. The
    # earlier version used a tok that returned "" for an empty decode and a model of
    # None, so summarise() returned "" whether the guard ran or not — unfalsifiable.
    from dume.models import Gate as _G

    class _Boom:
        def decode(self, t): raise AssertionError("summarise reached the model past its guard")
        def encode(self, s): return s.split()
    _gate = SimpleNamespace(tok=_Boom(), model=None)
    assert _G.summarise(_gate, [1, 2, 3], C.EXPERT_GEN_TOKENS - 1) == "", "map generated below the note floor"
    assert _G.summarise(_gate, [], C.TARGET_MAX_TOKENS) == "", "map generated for an empty input"
    # THE HEADER STAYS INSIDE THE RESERVE THE SCHEDULER ALREADY SPENT ON IT.
    # scheduler.py sizes span_max as free/slope - 2*TARGET_MAX_TOKENS on the stated
    # basis that the backpropped sequence is span + orientation header + generated
    # text. The old header was clamped at TARGET_MAX_TOKENS by construction; the
    # branch deleted that clamp, so summarise() has to carry it.
    assert "min(int(budget), C.TARGET_MAX_TOKENS)" in _iP.getsource(_G.summarise), \
        "the map can outgrow the memory the scheduler reserved for the header"
    # THE GATE FINISHES BEFORE THE POOL STARTS: split the input into prompts, THEN
    # activate k. Both paths go through ONE splitter, so what an expert is asked is
    # a function of the input and the law alone — never of how many seats were free.
    # Budgeting the map off the RESIDENT experts made it a function of residency:
    # the same row in a later epoch with a different seat count got a different map,
    # and the stationary context the frozen gate was chosen for quietly failed.
    # source assertions read CODE, never prose: these comments discuss summarise()
    # and residency at length, and a substring check over the raw text passes or
    # fails on the documentation rather than on what runs.
    import ast as _ast, textwrap as _tw

    def _code(fn) -> str:
        node = _ast.parse(_tw.dedent(_iP.getsource(fn))).body[0]
        for n in _ast.walk(node):                     # drop every docstring in the tree
            body = getattr(n, "body", None)
            if isinstance(body, list) and body and isinstance(body[0], _ast.Expr) \
                    and isinstance(getattr(body[0], "value", None), _ast.Constant) \
                    and isinstance(body[0].value.value, str):
                n.body = body[1:] or [_ast.Pass()]
        return _ast.unparse(node)                     # comments never survive the parse

    src_tr, src_an, src_sp = (_code(_tr_mod.System._train_one), _code(_tr_mod.System.answer),
                              _code(_tr_mod.System._split))
    assert "self.gate.summarise(plan.ids, min((x.n_tokens for x in" in src_sp, \
        "the map is not sized by the smallest span the law allocated"
    assert "plan.selections" in src_sp and "resident" not in src_sp, \
        "the split reads residency, so the prompt depends on what fitted in RAM"
    for name, src in (("training", src_tr), ("deployment", src_an)):
        assert "self._split(plan)" in src, f"{name} does not go through the splitter"
        assert src.index("self._split(plan)") < src.index("self.sched.ensure("), \
            f"{name} activates experts before the gate has split the input"
        assert "summarise" not in src, f"{name} generates the map inside the residency window"
    # ONE ENTRY PER FRAGMENT, NOT PER EXPERT. Deployment emits `cycles` Selections
    # sharing an eid and differing only in [start, end) — that is how an expert
    # covers a region wider than its span without paying a swap. Keyed by eid, the
    # last fragment's text would be handed to every fragment of that expert, and
    # the cycles would silently read the same slice k times.
    from dume.router import Selection as _Sel
    _sys = SimpleNamespace(gate=SimpleNamespace(tok=_Tok(), summarise=lambda ids, b: ""))
    _plan = SimpleNamespace(ids=[f"t{i}" for i in range(40)],
                            selections=[_Sel(eid=7, cid=0, start=0, end=10),
                                        _Sel(eid=7, cid=0, start=10, end=20),
                                        _Sel(eid=9, cid=1, start=20, end=40)])
    _cut = _tr_mod.System._split(_sys, _plan)
    assert len(_cut) == 3, f"a fragment was collapsed: {len(_cut)} of 3"
    assert [sp for _, sp, _ in _cut] == ["t0 t1 t2 t3 t4 t5 t6 t7 t8 t9",
                                         "t10 t11 t12 t13 t14 t15 t16 t17 t18 t19",
                                         " ".join(f"t{i}" for i in range(20, 40))], \
        "a fragment did not receive its own slice"
    assert len({cx for _, _, cx in _cut}) == 3, "two fragments share an orientation"
    print(f"router        OK  (anchors ordered, spans within input; training shares "
          f"{sorted(s.n_tokens for s in probe_spans)}; span_max clamps the share)")

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

    # thermal. The ordinal this all used to run on is inert: measured over 2900
    # archived batches and 541 live ones, NSProcessInfo returned `fair` every
    # time, so baseline, volatility and every interval derived from it were
    # identically zero and k_thermal sat at k_max for the whole run. The device
    # never cast a vote. Degrees off the die move 52 -> 67 C in the same minute.
    from dume.chain import ThermalRegulator
    rng = np.random.default_rng(0)

    def feed(temps, level=0.0):
        r = ThermalRegulator()
        for x in temps:
            r.observe(float(x), level)
        return r
    # a machine that idles, then works, then settles at its working temperature
    base = list(45 + rng.normal(0, 0.4, 60)) + list(52 + rng.normal(0, 0.6, 200))
    settled = feed(base)
    assert settled.peak - settled.floor > 5.0, "the fixture never established a span"

    # SETTLED IS FREE. Not "small": the jitter deadband is the machine's own
    # mean |z|, so its ripple costs nothing at all on most reads.
    calm = ThermalRegulator()
    ps = [calm.observe(float(x)) or calm.pressure() for x in base]
    tail = ps[120:]
    assert max(tail) < 0.1, f"a settled machine throttles itself: max p {max(tail):.4f}"
    assert sum(1 for x in tail if x > 0) < len(tail) // 4, \
        f"ripple priced on {sum(1 for x in tail if x > 0)}/{len(tail)} settled reads"
    assert ThermalRegulator().pressure() == 0.0, "a cold regulator invented pressure"
    assert ThermalRegulator().k_thermal(4.0) == 4.0, "a cold regulator throttled"

    # REACTION GROWS WITH THE EXCURSION, measured against the SPAN the machine
    # works over and not against its noise floor. Taken in the noise (mad
    # 0.33 C here) a 3 C rise scores 8.7 sigma, pressure 26.6, and k collapses
    # to 1.05 -- on a die that idles at 45 and works at 67.
    small, big = feed(base + [55.0]), feed(base + [60.0])
    assert 0.0 < small.pressure() < big.pressure(), \
        f"not monotone in excursion: {small.pressure():.3f} {big.pressure():.3f}"
    assert 2.5 < small.k_thermal(4.0) < 3.8, f"a 3 C rise gave k {small.k_thermal(4.0):.3f}"
    # the same absolute rise on a machine with a WIDER working range costs less
    wide = feed(list(30 + rng.normal(0, 0.4, 60)) + list(52 + rng.normal(0, 0.6, 200)) + [55.0])
    assert wide.peak - wide.floor > small.peak - small.floor
    assert wide.pressure() < small.pressure(), \
        f"span is not the scale: {wide.pressure():.4f} vs {small.pressure():.4f}"

    # z IS THE RATE TERM. Same temperature, different history: the mean lags, so
    # a die that has just arrived scores high and one that has been there scores
    # zero. This is why the explicit first and second derivatives were removed --
    # measured, they carried 0.8% and 0.6% of the signal.
    assert big.z > 0.4, f"a fresh 8 C rise did not register: z {big.z:+.4f}"
    assert abs(feed(base + [60.0] * 400).z) < 1e-9, "the normal never caught up"

    # COOLING IS FREE.
    cool = feed(base + [51.0, 50.0, 49.0, 48.0])
    assert cool.z < 0, "cooling did not read as cooling"
    assert cool.pressure() == 0.0 and cool.k_thermal(4.0) == 4.0, "a cooling machine was throttled"

    # THE NORMAL RE-LEARNS. Where a machine runs happily is a property of the
    # machine AND its room, and the room is not stationary.
    moved = feed(base + [60.0] * 400)
    assert moved.mean > 59.0, f"sustained new normal not re-learned: {moved.mean:.3f}"
    assert moved.pressure() == 0.0, f"still throttling its own normal: {moved.pressure():.4f}"
    assert abs(small.mean - settled.mean) < 0.1, \
        f"one spike moved the normal: {settled.mean:.4f} -> {small.mean:.4f}"

    # THE OS KEEPS A VETO UNDERNEATH. Apple's number, not ours, and by `serious`
    # the OS is already throttling, so adding experts worsens its complaint.
    assert feed(base, level=2.0).k_thermal(4.0) == 1.0, "nothing was out of hand at serious"
    assert feed(base, level=1.0).k_thermal(4.0) == settled.k_thermal(4.0), \
        "`fair` was treated as trouble"

    # no k value is baked in: k_max enters only as the base of the root.
    pr = small.pressure()
    for km in (4.0, 16.0):
        assert abs(small.k_thermal(km) - km ** (1.0 / (1.0 + pr))) < 1e-12, f"k_max ignored at {km}"

    # ONE READ PER BATCH. `Scheduler.k_thermal` advances the regulator, so every
    # quantity above is in units of "one read". It was read twice -- once by the
    # router and once by the health record -- which halved that clock.
    users = [l for f in ("dume/train.py", "dume/router.py")
             for l in open(f) if ".k_thermal" in l and "last_k_thermal" not in l]
    assert len(users) == 1, f"k_thermal read {len(users)}x outside the scheduler: {users}"

    print(f"thermal       OK  (die {die_temp():.1f} C off real sensors, span {settled.peak - settled.floor:.1f} C; "
          f"settled max p {max(tail):.3f}; +3 C -> k {small.k_thermal(4.0):.2f}, +8 C -> k {big.k_thermal(4.0):.2f}; "
          f"normal re-learned to {moved.mean:.1f} C; OS veto at serious)")
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
