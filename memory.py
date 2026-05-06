from __future__ import annotations
import hashlib
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import mlx.core as mx
import configs
from vectors import numpy_cosine_distance, compute_mean_inter_centroid_distance
@dataclass
class VoronoiCluster:
    cluster_id: str
    centroid: np.ndarray
    optimal_k: int
    top_experts: List[int]
    confidence: float
    sample_count: int
    r_out_snapshot: Dict[int, float]
    l_eff_scores: Dict[int, float]
    last_updated: int
    def update_confidence(self):
        self.confidence = min(1.0, self.sample_count / 50)
class RoutingMemory:
    def __init__(self):
        self.clusters: List[VoronoiCluster] = []
        self.tau: float = float('inf')
    def _to_numpy(self, mx_hidden: mx.array) -> np.ndarray:
        return np.array(mx_hidden.tolist(), dtype=np.float32)
    def _recompute_tau(self):
        if len(self.clusters) < 2:
            self.tau = float(configs.VORONOI_ALPHA)
            return
        centroids = [c.centroid for c in self.clusters]
        mean_dist = compute_mean_inter_centroid_distance(centroids)
        self.tau = max(1e-6, float(configs.VORONOI_ALPHA) * mean_dist)
    def get_dynamic_tau(self) -> float:
        return self.tau
    def lookup(self, gate_hidden: mx.array) -> Optional[VoronoiCluster]:
        if not self.clusters:
            return None
        vec = self._to_numpy(gate_hidden)
        best_dist = float('inf')
        best_cluster = None
        for cluster in self.clusters:
            dist = numpy_cosine_distance(vec, cluster.centroid)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster
        if best_cluster is not None and best_dist < self.tau:
            best_cluster.centroid = (
                configs.EMA_DECAY * best_cluster.centroid
                + (1 - configs.EMA_DECAY) * vec
            )
            norm = np.linalg.norm(best_cluster.centroid)
            if norm > 1e-8:
                best_cluster.centroid = best_cluster.centroid / norm
            best_cluster.sample_count += 1
            best_cluster.update_confidence()
            return best_cluster
        return None
    def spawn_cluster(self, gate_hidden, expert_ids, tkl_scores, r_out_snapshot, l_eff_scores, optimal_k, token_count, r_i, domain_mean_r_i):
        if r_i <= domain_mean_r_i:
            return None
        centroid = self._to_numpy(gate_hidden)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm
        cluster = VoronoiCluster(
            cluster_id=hashlib.sha256(centroid.tobytes()).hexdigest()[:16],
            centroid=centroid, optimal_k=optimal_k,
            top_experts=sorted(expert_ids, key=lambda i: tkl_scores.get(i, 0), reverse=True),
            confidence=min(1.0, 1 / 50), sample_count=1,
            r_out_snapshot=r_out_snapshot, l_eff_scores=l_eff_scores, last_updated=token_count,
        )
        self.clusters.append(cluster)
        self._recompute_tau()
        self._enforce_cluster_cap(token_count)
        return cluster
    def merge_close_clusters(self):
        merged = True
        while merged and len(self.clusters) > 1:
            merged = False
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    a, b = self.clusters[i], self.clusters[j]
                    dist = numpy_cosine_distance(a.centroid, b.centroid)
                    if dist < self.tau / 2:
                        total = a.sample_count + b.sample_count
                        mc = (a.sample_count * a.centroid + b.sample_count * b.centroid) / total
                        norm = np.linalg.norm(mc)
                        if norm > 1e-8:
                            mc = mc / norm
                        a.centroid = mc
                        a.sample_count = total
                        a.update_confidence()
                        if b.confidence > a.confidence:
                            a.top_experts = b.top_experts
                        self.clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break
        self._recompute_tau()
    def prune_stale(self, current_token_count: int):
        self.clusters = [
            c for c in self.clusters
            if not (current_token_count - c.last_updated > configs.CLUSTER_PRUNE_AGE and c.confidence < configs.CLUSTER_CONFIDENCE_FLOOR)
        ]
        self._recompute_tau()
    def _enforce_cluster_cap(self, token_count: int):
        cap = max(10, token_count // configs.CLUSTER_CAP_RATE)
        if len(self.clusters) > cap:
            self.clusters.sort(key=lambda c: c.confidence, reverse=True)
            self.clusters = self.clusters[:cap]
            self._recompute_tau()
    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self.clusters, f)
    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                self.clusters = pickle.load(f)
            self._recompute_tau()
        except Exception:
            self.clusters = []
    def sync(self, path: str):
        self.save(path)
class SessionTracker:
    def __init__(self):
        self.activations: Dict[int, List[dict]] = defaultdict(list)
        self.domain_tkl: Dict[str, List[float]] = defaultdict(list)
        self.domain_exposure: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.expert_tkl: Dict[int, float] = defaultdict(float)
        self.expert_domains: Dict[int, str] = {}
        self._expert_activations: Dict[int, int] = defaultdict(int)  # resets on migration
        self.token_count: int = 0
        self._timeline_a_tokens: int = 0
        self._warmup_logged: bool = False
    def record_activation(self, expert_id, tokens, r_i, wall_time, tkl_score=0.0, domain="general"):
        self.activations[expert_id].append({"tokens": tokens, "r_i": r_i, "wall_time": wall_time, "tkl": tkl_score})
        self.domain_tkl[domain].append(tkl_score)
        self.expert_tkl[expert_id] = tkl_score
        self.domain_exposure[expert_id][domain] += tokens
        self.expert_domains[expert_id] = domain
        self._expert_activations[expert_id] += 1
        self.token_count += tokens
    def record_timeline_a(self, tokens):
        self._timeline_a_tokens += tokens
        self.token_count += tokens
    def get_domain_mean_tkl(self, domain):
        scores = self.domain_tkl.get(domain, [])
        return float(np.mean(scores)) if scores else 0.0
    def get_domain_mean_score(self, domain):
        return self.get_domain_mean_tkl(domain)
    def get_expert_tkl(self, expert_id):
        return self.expert_tkl.get(expert_id, 0.0)
    def get_expert_activations(self, expert_id) -> int:
        return self._expert_activations.get(expert_id, 0)
    def get_domain_exposure(self, expert_id, domain):
        return self.domain_exposure[expert_id].get(domain, 0)
    def get_domain_mean_exposure(self, domain):
        exposures = [d.get(domain, 0) for d in self.domain_exposure.values() if domain in d]
        return int(np.mean(exposures)) if exposures else 0
    def get_total_tokens_seen(self):
        return self.token_count
    def get_current_allocation(self, expert_id):
        acts = self.activations.get(expert_id, [])
        return acts[-1].get("tokens", 0) if acts else 0
    def get_dominant_domain(self, expert_id):
        return self.expert_domains.get(expert_id, "general")
    def find_migration_target(self, expert_id, convolution):
        current = self.get_dominant_domain(expert_id)
        best_domain, best_score = "general", -1.0
        for domain, scores in self.domain_tkl.items():
            if domain == current:
                continue
            if scores:
                mean_score = float(np.mean(scores[-20:]))
                if mean_score > best_score:
                    best_score = mean_score
                    best_domain = domain
        return best_domain
    def record_migration(self, expert_id, new_domain):
        self.expert_domains[expert_id] = new_domain
        self.domain_exposure[expert_id] = defaultdict(int)
        self._expert_activations[expert_id] = 0  # reset cooldown counter
    def get_timeline_a_rate(self):
        return self._timeline_a_tokens / max(self.token_count, 1)
    def log_warmup(self, token_count):
        if not self._warmup_logged:
            if token_count < 500:
                print(f"[warmup] {token_count} tokens processed")
            else:
                print(f"[warmup] {token_count} tokens processed — warmup complete")
                self._warmup_logged = True
    def reset(self):
        self.activations.clear()
        self.domain_tkl.clear()
        self.domain_exposure.clear()
        self.expert_tkl.clear()
        self._timeline_a_tokens = 0
        self.token_count = 0
