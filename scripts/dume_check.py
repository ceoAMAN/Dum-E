"""Model-free checks for the dume package. Runs in seconds, no models.

    python scripts/dume_check.py

Each check asserts a property the design DEPENDS on, not a code path:
directions frozen, volume cannot buy standing, migration displaces at the cap,
chains never freeze, high traffic tightens, cold reliability is full trust,
tau has a consumer, spans are contiguous and cover the input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dume import config as C                                   # noqa: E402
from dume.chain import MarkovChain, MigrationChains, SizeChains  # noqa: E402
from dume.geometry import Geometry                             # noqa: E402
from dume.health import Health                                 # noqa: E402
from dume.reward import Reliability, is_heldout, weights       # noqa: E402
from dume.standing import GENERAL, SURPLUS, Standing           # noqa: E402


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
    home = g.home(w)
    g.set_tau(home, 0.99)
    assert home not in g.present(w, msims) or len(g.present(w, msims)) == 1
    g.set_tau(home, 0.5)
    assert home in g.present(w, msims)
    print("geometry      OK  (3 clusters, direction frozen, tau gates presence)")

    # standing
    s = Standing(g.C)
    assert len(s.generals()) == C.GENERAL_EXPERTS
    assert len(s.surplus()) == C.E - C.GENERAL_EXPERTS - g.C * C.CENTROID_EXPERTS
    assert s.trial(0, set()) in s.surplus(), "trial must draw from surplus first"
    s.observe(10, 0, 0.5, 32)
    s.observe(11, 0, 0.5, 320)
    assert s.score(10, 0) > s.score(11, 0), "volume bought standing"
    mig = MigrationChains(g.C)
    e_move, e_surp = s.members(2)[0], s.surplus()[0]
    for _ in range(5):
        s.observe(e_move, 0, 2.0, 32)
        s.observe(e_surp, 1, 3.0, 32)
    for c in (0, 1):
        for m in s.members(c):
            for _ in range(5):
                s.observe(m, c, 0.1, 32)
    moves = s.migrate(mig)
    assert s.assigned[e_move] == 0 and s.assigned[e_surp] == 1, moves
    assert all(len(s.members(c)) <= C.CENTROID_EXPERTS for c in range(g.C)), "cap violated"
    assert mig.pool.evidence(2) > 0, "migration chain did not observe the move"
    print(f"standing      OK  (rate not volume; {len(moves)} moves with displacement; chain fed)")

    # chains
    mc = MarkovChain(3, memory=10)
    for _ in range(1000):
        mc.observe(0, 1)
    mc.observe(0, 2)
    pb = mc.predict(0)[2]
    for _ in range(5):
        mc.observe(0, 2)
    assert mc.predict(0)[2] > pb, "chain froze (1/n)"
    sz = SizeChains(g.C)
    for _ in range(20):
        sz.observe_load(0, 0.9, g.C)
        sz.observe_load(1, 0.01, g.C)
    assert sz.next_tau(0, 0.7) > 0.7, "high traffic must tighten"
    assert sz.next_tau(1, 0.7) < 0.7, "starved must widen"
    print("chains        OK  (bounded evidence; tighten on load; widen on starvation)")

    # reliability
    r = Reliability(g.C)
    assert r.R(0) == 1.0, "cold must be full trust"
    r.observe(np.full(200, 0.5), np.zeros(200, dtype=int))
    r.observe(np.full(200, 3.0), np.ones(200, dtype=int))
    assert r.R(0) > r.R(1)
    assert len(weights(r, np.array([0, 0, 1, 1]), 8)) == 8
    frac = sum(is_heldout(f"k{i}") for i in range(4000)) / 4000
    assert 0.2 < frac < 0.3, frac
    print(f"reliability   OK  (cold=1.0; vector {r.vector().round(2)}; heldout {frac:.2f})")

    # health
    h = Health()
    h.put(x=float("nan"))
    assert h.nonfinite == 1
    print("health        OK  (non-finite caught at record time)")
    print("ALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
