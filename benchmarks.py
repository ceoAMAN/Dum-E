from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


class BenchmarkRecorder:
    """Single source of the paper's benchmark record, all measured live on the
    REAL training run. Buffers per-batch metrics; at each checkpoint it appends one
    consolidated row to logs/benchmarks.csv and refreshes logs/benchmarks_summary.json.

    Captures three groups:
      - Losses (every term): L_eff, L_dom, L_rel, the total L_gate, the expert MSE,
        and the L_div peer-similarity (lower = experts more diverged).
      - Audit A.2.4 proof (expert output reaches the reply): contribution_norm =
        ‖synthesis - base‖, the delta the experts cause in Central, plus r_i (their
        alignment with it). Both > 0 over the run = experts genuinely influence
        Central, i.e. the MoE pipeline is not a training-only apparatus.
      - Manual acceptance observables: K, per-domain K-velocity, expert weight
        divergence (std), routing-memory hit rate, cluster count, central
        reconstruction entropy, timeline-A rate, held-out gate accuracy.
    """

    COLUMNS = [
        "batch", "tokens",
        # ── losses (all terms) ──
        "gate_total", "l_eff", "l_dom", "l_rel", "expert_mse", "l_div_sim",
        # ── audit A.2.4: expert -> reply influence ──
        "r_i", "contribution_norm",
        # ── acceptance observables ──
        "k", "recon_entropy", "expert_weight_std", "cluster_hit_rate",
        "cluster_count", "timeline_a_rate", "gate_acc", "k_velocity_mean",
    ]

    def __init__(self, log_dir: str = "logs"):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv = self.dir / "benchmarks.csv"
        if not self.csv.exists():
            self.csv.write_text(",".join(self.COLUMNS) + "\n")
        self._buf: List[Dict[str, float]] = []
        self._cluster_hits = 0
        self._cluster_lookups = 0

    def record_batch(self, metrics: Dict[str, float]) -> None:
        """Buffer one timeline-B batch's metrics (keys are a subset of COLUMNS)."""
        self._buf.append(metrics)

    def note_cluster_lookup(self, hit: bool) -> None:
        self._cluster_lookups += 1
        if hit:
            self._cluster_hits += 1

    def cluster_hit_rate(self) -> float:
        return self._cluster_hits / max(1, self._cluster_lookups)

    def _recent_mean(self, key: str, n: int = 200) -> float:
        vals = [b[key] for b in self._buf[-n:] if key in b and b[key] == b[key]]
        return float(np.mean(vals)) if vals else 0.0

    def checkpoint(
        self,
        *,
        batch: int,
        tokens: int,
        cluster_count: int,
        timeline_a_rate: float,
        gate_acc: float,
        expert_weight_std: float,
        k_velocity_mean: float,
    ) -> Dict[str, float]:
        """Write one consolidated benchmark row (recent-window means) + refresh the
        summary json. Returns the row so the caller can print/inspect it."""
        row: Dict[str, float] = {
            "batch": batch,
            "tokens": tokens,
            "gate_total": self._recent_mean("gate_total"),
            "l_eff": self._recent_mean("l_eff"),
            "l_dom": self._recent_mean("l_dom"),
            "l_rel": self._recent_mean("l_rel"),
            "expert_mse": self._recent_mean("expert_mse"),
            "l_div_sim": self._recent_mean("l_div_sim"),
            "r_i": self._recent_mean("r_i"),
            "contribution_norm": self._recent_mean("contribution_norm"),
            "k": self._recent_mean("k"),
            "recon_entropy": self._recent_mean("recon_entropy"),
            "expert_weight_std": expert_weight_std,
            "cluster_hit_rate": self.cluster_hit_rate(),
            "cluster_count": cluster_count,
            "timeline_a_rate": timeline_a_rate,
            "gate_acc": gate_acc,
            "k_velocity_mean": k_velocity_mean,
        }
        with open(self.csv, "a") as f:
            f.write(",".join(self._fmt(row[c]) for c in self.COLUMNS) + "\n")
        with open(self.dir / "benchmarks_summary.json", "w") as f:
            json.dump(row, f, indent=2)
        return row

    @staticmethod
    def _fmt(v) -> str:
        return f"{v:.6f}" if isinstance(v, float) else str(v)
