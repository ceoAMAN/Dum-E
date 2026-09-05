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
    domain: str = "general"
    def update_confidence(self):
        self.confidence = min(1.0, self.sample_count / 50)
class RoutingMemory:
    def __init__(self):
        self.clusters: List[VoronoiCluster] = []
        self.tau: float = float('inf')
    def _to_numpy(self, mx_hidden: mx.array) -> np.ndarray:
        return np.array(mx_hidden.tolist(), dtype=np.float32)
    def _recompute_tau(self):
        # Cold cache (<2 clusters): use a tight absolute threshold, not the old
        # VORONOI_ALPHA fallback (0.30) which let the first cluster falsely match
        # almost any prompt. VORONOI_TAU_COLD sits inside the measured
        # same/different-intent separation band.
        if len(self.clusters) < 2:
            self.tau = float(configs.VORONOI_TAU_COLD)
            return
        centroids = [c.centroid for c in self.clusters]
        mean_dist = compute_mean_inter_centroid_distance(centroids)
        # Warm: scale by mean inter-centroid spread, but cap so a diverse cache
        # can't loosen tau past the point where unrelated prompts start matching.
        self.tau = max(1e-6, min(float(configs.VORONOI_TAU_CEIL),
                                 float(configs.VORONOI_ALPHA) * mean_dist))
    def get_dynamic_tau(self) -> float:
        return self.tau
    def get_domain_mean_k(self) -> int:
        ks = [int(cluster.optimal_k) for cluster in self.clusters if int(cluster.optimal_k) > 0]
        return round(sum(ks) / len(ks)) if ks else 1
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
    def spawn_cluster(self, gate_hidden, expert_ids, tkl_scores, r_out_snapshot, l_eff_scores, optimal_k, token_count, r_i, domain_mean_r_i, domain="general"):
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
            domain=domain,
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
                    # Merge at the MEMBER band, not tau/2. Tau is calibrated
                    # query-to-query (paraphrase 0.018 / unrelated 0.136); this
                    # comparison is centroid-to-centroid, a different population.
                    # tau/2 = 0.020 means cot 4.92, and the largest cot anywhere
                    # in the stored clusters is 4.58 — the old threshold sat
                    # above the ceiling of the data, so 0 of 46 same-domain
                    # pairs could ever qualify and no merge was possible.
                    if dist <= 1.0 - configs.SIM_MEMBER:
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
                            a.domain = getattr(b, "domain", getattr(a, "domain", "general"))
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
        self._last_warmup_log_tokens: int = -1
    def record_activation(self, expert_id, tokens, r_i, wall_time, tkl_score=0.0, domain="general",
                          disagreement=None, novel=False):
        """`disagreement` in [0,1] is how far this expert's output sat from its
        peers on the same input — the hallucination proxy (an expert alone in
        left field is the classic confabulation signature). `novel` marks a batch
        the routing cache had never seen, so adaptability can be scored
        separately from performance on familiar ground."""
        self.activations[expert_id].append({
            "tokens": tokens, "r_i": r_i, "wall_time": wall_time, "tkl": tkl_score,
            "disagreement": disagreement, "novel": bool(novel), "domain": domain,
        })
        self.domain_tkl[domain].append(tkl_score)
        self.expert_tkl[expert_id] = tkl_score
        self.domain_exposure[expert_id][domain] += tokens
        self.expert_domains[expert_id] = domain
        self._expert_activations[expert_id] += 1
        self.token_count += tokens
        # Bounded history. Unbounded lists would grow forever now that sessions
        # no longer wipe them, and an unbounded mean is exactly the 1/n freeze:
        # observation 10_000 would move a score by 0.01% and the expert would
        # stop responding to its own recent behaviour with no symptom.
        cap = int(configs.SESSION_HISTORY_CAP)
        if len(self.activations[expert_id]) > cap:
            del self.activations[expert_id][:-cap]
        if len(self.domain_tkl[domain]) > cap:
            del self.domain_tkl[domain][:-cap]

    def composite_tkl(self, expert_id: int, domain: Optional[str] = None) -> Dict[str, float]:
        """The full Triple-K rating: a weighted sum of everything we can measure
        about an expert, replacing `r_out * (r_i / c_e) * anchor` — which mixed
        tokens², a quality score and seconds into one number that allocation size
        dominated.

        Every component is normalised to [0, 1] so the sum is a real weighted
        average rather than whichever term happens to carry the largest units:

          domain_perf   mean r_i on `domain` (relevance-weighted)
          processing    throughput, tokens/sec, scaled against this expert's best
          output_time   inverse latency (fast is better)
          no_halluc     1 - mean disagreement with peers on the same input
          correctness   mean r_i overall (grounded r_i when the caller supplies it)
          learning      slope of r_i over its recent history (is it improving?)
          novelty       mean r_i on batches the routing cache had never seen

        Components with no data yet are dropped rather than counted as zero — a
        missing measurement must not read as a bad one. Returns the breakdown as
        well as `overall`, so ranking can use domain relevance, domain
        performance, or the composite."""
        acts = self.activations.get(expert_id, [])
        if not acts:
            return {"overall": 0.0, "domain_perf": 0.0, "n": 0}
        out: Dict[str, float] = {}
        r_all = [a["r_i"] for a in acts if a.get("r_i") is not None]
        in_dom = [a for a in acts if domain is None or a.get("domain") == domain]
        r_dom = [a["r_i"] for a in in_dom if a.get("r_i") is not None]
        if r_dom:
            out["domain_perf"] = float(np.clip(np.mean(r_dom), 0.0, 1.0))
        if r_all:
            out["correctness"] = float(np.clip(np.mean(r_all), 0.0, 1.0))
        tps = [a["tokens"] / a["wall_time"] for a in acts if a.get("wall_time", 0) > 0]
        if tps:
            best = max(tps)
            out["processing"] = float(np.clip(np.mean(tps) / best, 0.0, 1.0)) if best > 0 else 0.0
            wt = [a["wall_time"] for a in acts if a.get("wall_time", 0) > 0]
            fastest = min(wt)
            out["output_time"] = float(np.clip(fastest / float(np.mean(wt)), 0.0, 1.0))
        dis = [a["disagreement"] for a in acts if a.get("disagreement") is not None]
        if dis:
            out["no_halluc"] = float(np.clip(1.0 - np.mean(dis), 0.0, 1.0))
        if len(r_all) >= 3:
            # Learning rate: is r_i trending up? Slope over the recent window,
            # squashed to [0,1] with 0.5 == flat, so improving beats plateaued.
            recent = r_all[-20:]
            x = np.arange(len(recent), dtype=np.float64)
            slope = float(np.polyfit(x, np.array(recent, dtype=np.float64), 1)[0])
            out["learning"] = float(np.clip(0.5 + 10.0 * slope, 0.0, 1.0))
        nov = [a["r_i"] for a in acts if a.get("novel") and a.get("r_i") is not None]
        if nov:
            out["novelty"] = float(np.clip(np.mean(nov), 0.0, 1.0))
        out["overall"] = float(np.mean(list(out.values()))) if out else 0.0
        out["n"] = len(acts)
        return out

    def composite_tkl_pool(self, expert_ids: List[int], domain: Optional[str] = None) -> Dict[int, Dict[str, float]]:
        """Composite TKL for a set of experts, SHRUNK toward the pool mean by
        evidence count. Ranking is inherently a pool operation, and without
        shrinkage an expert measured twice can outrank one measured fifty times:
        dropping its unmeasured components leaves only the favourable ones, so
        thin evidence reads as excellence. (Observed directly: 0.800 on n=2
        beating 0.764 on n=12.)

            shrunk = (n*raw + m*pool_mean) / (n + m),   m = median n across the pool

        The pseudo-count m is the pool's own median activation count, so nothing
        is hardcoded and the shrinkage relaxes automatically as evidence
        accumulates — at n >> m the raw score is returned essentially untouched."""
        raw = {e: self.composite_tkl(e, domain) for e in expert_ids}
        scored = [(e, c) for e, c in raw.items() if c.get("n", 0) > 0]
        if not scored:
            return raw
        # WEIGHTED sum, with the weights DERIVED rather than declared: each
        # component is weighted by its spread across the pool. A component on
        # which every expert scores the same separates nobody, so it contributes
        # nothing and stops diluting the score; a component that spreads experts
        # apart is the one actually carrying the ranking information. An
        # unweighted mean gives a flat component the same say as a decisive one.
        parts = [k for k in ("domain_perf", "correctness", "processing",
                             "output_time", "no_halluc", "learning", "novelty")]
        spread: Dict[str, float] = {}
        for k in parts:
            vals = [c[k] for _, c in scored if k in c]
            spread[k] = float(np.std(vals)) if len(vals) > 1 else 0.0
        total_spread = sum(spread.values())
        for e, c in scored:
            have = [k for k in parts if k in c]
            if not have:
                continue
            if total_spread > 1e-9:
                w = {k: spread[k] for k in have}
                wsum = sum(w.values())
                if wsum > 1e-9:
                    c["overall"] = float(sum(w[k] * c[k] for k in have) / wsum)
                    c["weights"] = {k: round(w[k] / wsum, 4) for k in have}
            # else: every component flat across the pool -> the unweighted mean
            # from composite_tkl stands, since no component has a claim to more.
        pool_mean = float(np.mean([c["overall"] for _, c in scored]))
        m = float(np.median([c["n"] for _, c in scored])) or 1.0
        for e, c in scored:
            n = float(c["n"])
            c["overall_raw"] = c["overall"]
            c["overall"] = (n * c["overall_raw"] + m * pool_mean) / (n + m)
            c["evidence"] = n / (n + m)          # 0 = pure prior, 1 = fully self-determined
        return raw
    def record_timeline_a(self, tokens):
        self._timeline_a_tokens += tokens
        self.token_count += tokens
    def tkl_rank(self, expert_ids: List[int], domain: Optional[str] = None,
                 assigned: Optional[Dict[int, str]] = None) -> List[int]:
        """Rank experts for `domain`, best first, LEXICOGRAPHICALLY:

            1. domain relevance   — is this expert actually assigned here
            2. domain performance — its mean score in this domain
            3. overall            — the evidence-shrunk composite

        Three levels, not one blended number: an expert assigned to the domain
        outranks a visitor even if the visitor's overall is higher, because
        relevance is the first question. Only within equal relevance does domain
        performance decide, and only within equal domain performance does the
        composite break the tie."""
        pool = self.composite_tkl_pool(expert_ids, domain)
        assigned = assigned or {}

        def key(e: int):
            c = pool.get(e, {})
            relevant = 1 if (domain is not None and assigned.get(e) == domain) else 0
            return (relevant, c.get("domain_perf", 0.0), c.get("overall", 0.0))

        return sorted(expert_ids, key=key, reverse=True)

    def composite_domain_mean(self, expert_ids: List[int], domain: Optional[str] = None) -> float:
        """Mean COMPOSITE score across these experts — the like-for-like partner to
        get_expert_tkl now that expert_tkl holds the composite. Comparing a
        composite (0..1) against get_domain_mean_tkl (legacy, floored at 32) makes
        every expert look starved by construction."""
        pool = self.composite_tkl_pool(expert_ids, domain)
        vals = [c["overall"] for c in pool.values() if c.get("n")]
        return float(np.mean(vals)) if vals else 0.0

    def get_domain_mean_tkl(self, domain):
        scores = self.domain_tkl.get(domain, [])
        return float(np.mean(scores)) if scores else 0.0
    def get_domain_mean_score(self, domain):
        return self.get_domain_mean_tkl(domain)

    def get_domain_mean_r_i(self, domain: Optional[str] = None) -> float:
        """Mean r_i over recorded activations in this domain.

        spawn_cluster gates on `r_i > domain_mean_r_i`, and it must be compared
        against an r_i — get_domain_mean_tkl returns a TKL, which lives on a
        different scale entirely (floored at 32 in the legacy path), so comparing
        the two makes every batch look either exceptional or hopeless by
        construction rather than by measurement."""
        vals = [a["r_i"] for acts in self.activations.values() for a in acts
                if a.get("r_i") is not None and (domain is None or a.get("domain") == domain)]
        return float(np.mean(vals)) if vals else 0.0
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
        if self._warmup_logged:
            return
        if token_count >= 500:
            print(f"[warmup] {token_count} tokens processed — warmup complete")
            self._warmup_logged = True
        elif token_count > 0 and self._last_warmup_log_tokens <= 0:
            # Single boot confirmation when the first tokens are processed
            print(f"[warmup] {token_count} tokens processed — warming up")
            self._last_warmup_log_tokens = token_count
    def reset(self):
        """End-of-session reset. Clears the per-session COUNTERS only.

        This used to clear activations, domain_tkl, domain_exposure and
        expert_tkl as well. That was deliberate once — experts were being
        retrained constantly and stale scores were worse than none — but it means
        every expert's measured standing died at session end, get_current_allocation
        always returned 0, and nothing could accumulate across runs. Expert
        history is now the substrate for class standing and migration, so it
        survives. Use reset_history() for a genuine wipe."""
        self._timeline_a_tokens = 0
        # token_count is NOT reset: it is the monotonic clock cluster ages are
        # measured against, so zeroing it per session makes every stored
        # cluster look like it came from the future.

    def reset_history(self):
        """Genuine wipe. Only for a deliberate cold start — the stored history is
        what class standing, migration and staleness are all computed from."""
        self.activations.clear()
        self.domain_tkl.clear()
        self.domain_exposure.clear()
        self.expert_tkl.clear()
        self.expert_domains.clear()
        self._expert_activations.clear()
        self.token_count = 0
        self.reset()

    def save(self, path: str):
        """Plain-dict serialisation: domain_exposure is a defaultdict built from
        a lambda, which pickle cannot handle, so the nesting is flattened here
        rather than relying on the container type surviving."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "activations": {int(k): list(v) for k, v in self.activations.items()},
            "domain_tkl": {str(k): list(v) for k, v in self.domain_tkl.items()},
            "domain_exposure": {int(k): dict(v) for k, v in self.domain_exposure.items()},
            "expert_tkl": {int(k): float(v) for k, v in self.expert_tkl.items()},
            "expert_domains": dict(self.expert_domains),
            "_expert_activations": {int(k): int(v) for k, v in self._expert_activations.items()},
            # token_count is the CLOCK that VoronoiCluster.last_updated is stamped
            # against. Dropping it here made it restart at 0 every process, so
            # prune_stale computed a NEGATIVE age for every cluster carried over
            # from a previous run and could never prune one — sub-floor-confidence
            # clusters on disk became immortal.
            "token_count": int(self.token_count),
            "_warmup_logged": bool(self._warmup_logged),
            "_last_warmup_log_tokens": int(self._last_warmup_log_tokens),
        }
        with open(p, "wb") as f:
            pickle.dump(blob, f)

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                blob = pickle.load(f)
        except Exception as e:
            print(f"[warn] session tracker load failed ({e}); starting empty")
            return
        if not isinstance(blob, dict):
            print(f"[warn] session tracker at {path} is a {type(blob).__name__}, not a state "
                  f"dict (older format?); starting empty")
            return
        try:
            cap = int(configs.SESSION_HISTORY_CAP)
            for k, v in blob.get("activations", {}).items():
                # Apply the cap on LOAD too. It was enforced only in
                # record_activation, so a file written by an uncapped version
                # restored at full length and stayed that way until that expert
                # happened to be activated again.
                self.activations[int(k)] = list(v)[-cap:] if cap > 0 else list(v)
            for k, v in blob.get("domain_tkl", {}).items():
                self.domain_tkl[str(k)] = list(v)[-cap:] if cap > 0 else list(v)
            for k, v in blob.get("domain_exposure", {}).items():
                for d, n in dict(v).items():
                    self.domain_exposure[int(k)][str(d)] = int(n)
            self.expert_tkl.update({int(k): float(v) for k, v in blob.get("expert_tkl", {}).items()})
            self.expert_domains.update(blob.get("expert_domains", {}))
            self._expert_activations.update({int(k): int(v) for k, v in blob.get("_expert_activations", {}).items()})
            self.token_count = int(blob.get("token_count", 0))
            self._warmup_logged = bool(blob.get("_warmup_logged", False))
            self._last_warmup_log_tokens = int(blob.get("_last_warmup_log_tokens", -1))
        except Exception as e:
            # The unpickle succeeded but the SHAPE was wrong. Only the open/load
            # was guarded before, so a structurally different blob crashed boot
            # instead of falling back to an empty tracker.
            print(f"[warn] session tracker at {path} has an unexpected shape ({e}); "
                  f"starting empty")
            self.reset_history()
            return
        n_exp = len(self.activations)
        n_act = sum(len(v) for v in self.activations.values())
        print(f"[boot] session tracker: {n_act} activations across {n_exp} experts restored")
