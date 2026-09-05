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

from typing import Dict, List, Tuple

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
    def __init__(self, B: np.ndarray, tau: np.ndarray, version: Dict[str, object]):
        self.B = _unit(np.asarray(B, dtype=np.float32))          # (C, D) FROZEN
        self.tau = np.asarray(tau, dtype=np.float32).copy()       # (C,)   breathes
        self.version = dict(version)
        self.pair_sim = self.B @ self.B.T

    @property
    def C(self) -> int:
        return int(self.B.shape[0])

    # ── formation (offline, once) ───────────────────────────────────────────
    @staticmethod
    def form(X: np.ndarray, n_clusters: int, version: Dict[str, object]) -> "Geometry":
        """Max-min seeded cosine k-means on unit vectors. Seeds are the EXTREMES
        (Aman p.4: 'it takes extremes'); the centroid is the normalised mean —
        'the M vector close to mostly all vectors'. Shared directions are merged,
        and tau is calibrated LAST, against the members the frozen directions
        actually have."""
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
        # merge shared directions
        keep: List[int] = []
        for c in range(k):
            if all(float(B[c] @ B[j]) < C.SIM_MEMBER for j in keep):
                keep.append(c)
        B = B[keep]
        assign = np.argmax(X @ B.T, axis=1)
        # calibrate tau LAST
        tau = np.zeros(len(keep), dtype=np.float32)
        for c in range(len(keep)):
            sims = (X @ B[c])[assign == c]
            tau[c] = max(C.SIM_FAR, float(np.percentile(sims, 10))) if len(sims) else C.SIM_NEIGHBOUR
            tau[c] = min(tau[c], C.TAU_MAX)
        g = Geometry(B, tau, {**version, "n_vectors": int(N), "n_clusters": int(len(keep))})
        print(f"[geometry] formed {g.C} clusters from {N} vectors; tau in "
              f"[{tau.min():.3f}, {tau.max():.3f}]; version {g.version}")
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

    def present(self, w: np.ndarray, mean_sims: np.ndarray) -> List[int]:
        """Clusters whose TERRITORY contains this input — sim >= tau_c — ordered
        by composition weight. This is tau's consumer: tightening a cluster
        removes it from inputs at its edge, and the traffic lands on neighbours.
        Falls back to the nearest centroid so every input routes somewhere."""
        order = np.argsort(-w)
        out = [int(c) for c in order if float(mean_sims[c]) >= float(self.tau[c])]
        return out or [int(np.argmax(mean_sims))]

    def tier(self, home: int, c: int) -> str:
        if c == home:
            return "member"
        s = float(self.pair_sim[home, c])
        if s >= C.SIM_NEIGHBOUR:
            return "neighbour"
        if s >= C.SIM_FAR:
            return "close"
        return "far"

    def in_territory(self, v: np.ndarray, c: int) -> bool:
        return float(_unit(v) @ self.B[c]) >= float(self.tau[c])

    # ── the one online write ────────────────────────────────────────────────
    def set_tau(self, c: int, tau: float) -> None:
        self.tau[c] = float(max(C.SIM_FAR, min(C.TAU_MAX, tau)))

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, object]:
        return {"B": self.B, "tau": self.tau, "version": self.version}

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "Geometry":
        return Geometry(d["B"], d["tau"], d["version"])
