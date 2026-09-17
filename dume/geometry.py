"""Cluster geometry. Directions are formed OFFLINE and frozen; only tau breathes.

A cluster is a spherical cap {x : v_k . x >= tau_k}. Its area is monotone in
tau_k alone, so "resizing a cluster" is one number, and that number is the only
thing anything online is allowed to write (rule 15). The centroid array is
written exactly once, by form(), and stamped with the version of everything
that produced it (rule 16): a different gate, extractor or corpus invalidates it
and the loader says so out loud.

Token type (Aman, notebook p.10): "the centroid thing we get which gating
identifies with basis dot product" — each token's hidden state is assigned to
the centroid it is closest to. That assignment is what the reliability vector is
indexed by.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C


def _unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-8)


def nnls_weights(B: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Non-negative least squares of v onto the centroid rows of B, normalised to
    sum 1. This is the COMPOSITION of an input: how much of each centroid it is."""
    from scipy.optimize import nnls
    w, _ = nnls(B.T, v)
    s = w.sum()
    return (w / s) if s > 1e-12 else np.full(B.shape[0], 1.0 / B.shape[0])


class Geometry:
    def __init__(self, B: np.ndarray, tau: np.ndarray, version: Dict[str, object],
                 size: Optional[np.ndarray] = None, seen: Optional[np.ndarray] = None):
        self.B = _unit(np.asarray(B, dtype=np.float32))          # (C, D) FROZEN
        self.tau = np.asarray(tau, dtype=np.float32).copy()       # (C,)   breathes
        self.version = dict(version)
        self.pair_sim = self.B @ self.B.T
        # membership at formation. FROZEN with the directions: this is domain rank,
        # and it must not be re-derived from live traffic, because tau breathes
        # toward EQUAL presence and would drive any live estimate to uniform.
        self.size = (np.full(self.C, 1.0, dtype=np.float64) if size is None
                     else np.asarray(size, dtype=np.float64).copy())
        # LIVE input accumulated per centroid. Unlike size (frozen, = domain rank)
        # this one grows, and it is the only thing that grows: it buys a centroid
        # SEATS, not rank. A centroid that keeps receiving input needs more
        # experts to serve it.
        self.seen = (np.zeros(self.C, dtype=np.float64) if seen is None
                     else np.asarray(seen, dtype=np.float64).copy())

    @property
    def C(self) -> int:
        return int(self.B.shape[0])

    # ── formation (offline, once) ───────────────────────────────────────────
    @staticmethod
    def form(X: np.ndarray, n_clusters: int, version: Dict[str, object]) -> "Geometry":
        """Max-min seeded cosine k-means on unit vectors. Seeds are the EXTREMES
        (Aman p.4: 'it takes extremes'); the centroid is the normalised mean —
        'the M vector close to mostly all vectors'. Shared directions are merged
        (the larger membership survives), clusters too small to calibrate a tau
        are dropped and their members reassigned, and tau is calibrated LAST,
        against the members the frozen directions actually have."""
        X = _unit(np.asarray(X, dtype=np.float32))
        N = X.shape[0]
        k = max(1, min(int(n_clusters), N))
        idx = [int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))]
        while len(idx) < k:
            maxsim = (X @ X[idx].T).max(axis=1)
            maxsim[idx] = np.inf
            idx.append(int(np.argmin(maxsim)))
        B = X[idx].copy()
        assign = np.zeros(N, dtype=np.int64)
        for _ in range(25):
            assign = np.argmax(X @ B.T, axis=1)
            newB = B.copy()
            for c in range(k):
                m = X[assign == c]
                if len(m):
                    newB[c] = _unit(m.mean(0))
            if np.allclose(newB, B, atol=1e-5):
                break
            B = newB
        dropped = 0
        while True:
            assign = np.argmax(X @ B.T, axis=1)
            counts = np.bincount(assign, minlength=len(B))
            # merge shared directions, largest membership first so it survives
            keep: List[int] = []
            for c in np.argsort(-counts):
                if all(float(B[c] @ B[j]) < C.SIM_MEMBER for j in keep):
                    keep.append(int(c))
            B = B[sorted(keep)]
            assign = np.argmax(X @ B.T, axis=1)
            counts = np.bincount(assign, minlength=len(B))
            small = counts < C.MIN_MEMBERS
            if len(B) > 1 and small.any():      # max-min seeding is outlier-seeking; a
                dropped += int(small.sum())      # singleton cluster has no tau and no traffic
                B = B[~small]
                continue
            break
        # calibrate tau LAST
        tau = np.zeros(len(B), dtype=np.float32)
        for c in range(len(B)):
            sims = (X @ B[c])[assign == c]
            tau[c] = max(C.SIM_FAR, float(np.percentile(sims, C.TAU_PERCENTILE))) if len(sims) else C.SIM_NEIGHBOUR
            tau[c] = min(tau[c], C.TAU_MAX)
        counts = np.bincount(assign, minlength=len(B)).astype(np.float64)
        g = Geometry(B, tau, {**version, "n_vectors": int(N), "n_clusters": int(len(B))}, size=counts)
        print(f"[geometry] formed {g.C} clusters from {N} vectors (dropped {dropped} below "
              f"{C.MIN_MEMBERS} members); sizes {counts.astype(int).tolist()}; "
              f"tau in [{tau.min():.3f}, {tau.max():.3f}]; general=c{g.general()}; version {g.version}")
        return g

    # ── reading ─────────────────────────────────────────────────────────────
    def compose(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(w, assign, sims, mean_sims): composition weights (C,), per-token
        centroid (T,), per-token similarities (T, C), and the input mean's
        similarity to every centroid (C,) — the number the territory test reads."""
        Hu = _unit(np.asarray(H, dtype=np.float32))
        sims = Hu @ self.B.T
        assign = np.argmax(sims, axis=1)
        m = _unit(Hu.mean(0))
        w = nnls_weights(self.B, m)
        return w, assign, sims, (self.B @ m)

    def assign_tokens(self, H: np.ndarray) -> np.ndarray:
        return np.argmax(_unit(np.asarray(H, dtype=np.float32)) @ self.B.T, axis=1)

    def home(self, w: np.ndarray) -> int:
        return int(np.argmax(w))

    def inside(self, mean_sims: np.ndarray) -> List[int]:
        """Clusters whose TERRITORY contains this input: sim >= tau_c. This is
        the quantity tau gates and the one the size chains measure."""
        return [c for c in range(self.C) if float(mean_sims[c]) >= float(self.tau[c])]

    def present(self, w: np.ndarray, mean_sims: np.ndarray) -> List[int]:
        """inside(), ordered by composition weight. Tightening a cluster removes
        it from inputs at its edge and the traffic lands on neighbours. Falls
        back to the nearest centroid so every input routes somewhere."""
        ins = set(self.inside(mean_sims))
        out = [int(c) for c in np.argsort(-w) if int(c) in ins]
        return out or [int(np.argmax(mean_sims))]

    def domain(self) -> np.ndarray:
        """D[c] = size[c] / sum(size). Domain rank: how much of the corpus lives
        in this cluster. The weight the overall expert ranking sums over."""
        tot = float(self.size.sum())
        return (self.size / tot) if tot > 0 else np.full(self.C, 1.0 / self.C)

    def grow(self, inside: List[int]) -> None:
        """One input landed in each of these centroids. The ONLY writer of seen."""
        for c in inside:
            if 0 <= int(c) < self.C:
                self.seen[int(c)] += 1.0

    def capacity(self) -> np.ndarray:
        """Seats each centroid has EARNED: floor(sqrt(seen)), ceilinged by
        CENTROID_EXPERTS. The same sqrt bracket k_tier and GENERAL_EXPERTS use.

        A cold centroid has 0 seats, so every expert is general until a centroid
        has accumulated enough input to need one — 1 seat at 1 input, 2 at 4, 9
        at 81, and never more than 9 however much arrives. That ceiling is the
        anti-dominance constant: growth buys depth up to a bound, never the pool."""
        return np.minimum(np.floor(np.sqrt(self.seen)), C.CENTROID_EXPERTS).astype(np.int64)

    def band(self) -> Tuple[int, int]:
        """The sqrt(C) admissibility band on how many centroids may be live at
        once. A perfect square gets [sqrt-1, sqrt+1]; otherwise the predecessor
        and successor of sqrt(C), which is NOT +-1. Keeps the live domain count
        from swinging — the same defensive posture as ALLOC's 0 < beta <= 1."""
        r = math.sqrt(self.C)
        n = int(round(r))
        return (max(1, n - 1), n + 1) if n * n == self.C else (max(1, int(math.floor(r))), int(math.ceil(r)))

    def general(self) -> int:
        """The general centroid: the one sitting in the centre, relating
        maximally to all the others. MEASURED off pair_sim, never synthesised —
        it emerges from formation as the data grows, it is not a direction we
        construct."""
        if self.C == 1:
            return 0
        off = self.pair_sim.astype(np.float64).copy()
        np.fill_diagonal(off, np.nan)
        return int(np.nanargmax(np.nanmean(off, axis=1)))

    def tier(self, home: int, c: int) -> str:
        """How cluster `c` stands to this input's home, on the 10:20:30:40 bands
        over the frozen pair-similarity matrix. A DEPLOYMENT primitive (Aman,
        2026-09-18): training does not consult it, because training sweeps the
        pool for equal exposure and must not prefer one cluster's experts over
        another's — that is exactly the concentration the curriculum exists to
        avoid. It is a pure query on the geometry, so it costs nothing when
        uncalled; it currently has NO consumer, since answer() orders notes by
        standing alone.

        Distinct from the EXPERT tier (standing / membership): this one is about
        the distance between centroids, that one about how well an expert does.
        Two tier systems, deliberately separate."""
        if c == home:
            return "member"
        s = float(self.pair_sim[home, c])
        if s >= C.SIM_NEIGHBOUR:
            return "neighbour"
        if s >= C.SIM_FAR:
            return "close"
        return "far"

    # ── the one online write ────────────────────────────────────────────────
    def set_tau(self, c: int, tau: float) -> None:
        self.tau[c] = float(max(C.SIM_FAR, min(C.TAU_MAX, tau)))

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, object]:
        return {"B": self.B, "tau": self.tau, "version": self.version, "size": self.size,
                "seen": self.seen}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Geometry":
        return Geometry(d["B"], d["tau"], d["version"], size=d.get("size"), seen=d.get("seen"))
