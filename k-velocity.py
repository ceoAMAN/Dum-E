from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import configs


@dataclass
class UsageEvent:
    timestamp: float
    expert_ids: List[int]
    domain: str
    k_used: int
    confidence: float
    mean_r_i: float
    user_query_hash: int


@dataclass
class ExpertProfile:
    expert_id: int
    total_activations: int = 0
    success_activations: int = 0
    domain_affinities: Dict[str, float] = field(default_factory=dict)
    time_of_day_weights: List[float] = field(default_factory=lambda: [1.0] * 24)
    last_activated: float = 0.0
    velocity: float = 0.0


@dataclass
class DomainPattern:
    domain: str
    hour_distribution: List[float] = field(default_factory=lambda: [0.0] * 24)
    avg_k: float = 4.0
    preferred_experts: List[int] = field(default_factory=list)
    total_queries: int = 0


class KVelocity:
    def __init__(
        self,
        decay_rate: float = 0.995,
        learning_rate: float = 0.01,
        history_file: Optional[str] = None,
    ):
        self._decay_rate = decay_rate
        self._lr = learning_rate
        self._history_file = Path(history_file or "state/k_velocity.json")
        self._events: deque[UsageEvent] = deque(maxlen=10000)
        self._expert_profiles: Dict[int, ExpertProfile] = {}
        self._domain_patterns: Dict[str, DomainPattern] = {}
        self._query_domain_cache: Dict[int, str] = {}
        self._session_start = time.time()
        self._load()

    def _load(self):
        if not self._history_file.exists():
            return
        try:
            data = json.loads(self._history_file.read_text())
            for ep_data in data.get("expert_profiles", []):
                ep = ExpertProfile(
                    expert_id=ep_data["expert_id"],
                    total_activations=ep_data.get("total_activations", 0),
                    success_activations=ep_data.get("success_activations", 0),
                    domain_affinities=ep_data.get("domain_affinities", {}),
                    time_of_day_weights=ep_data.get("time_of_day_weights", [1.0] * 24),
                    last_activated=ep_data.get("last_activated", 0.0),
                    velocity=ep_data.get("velocity", 0.0),
                )
                self._expert_profiles[ep.expert_id] = ep
            for dp_data in data.get("domain_patterns", []):
                dp = DomainPattern(
                    domain=dp_data["domain"],
                    hour_distribution=dp_data.get("hour_distribution", [0.0] * 24),
                    avg_k=dp_data.get("avg_k", 4.0),
                    preferred_experts=dp_data.get("preferred_experts", []),
                    total_queries=dp_data.get("total_queries", 0),
                )
                self._domain_patterns[dp.domain] = dp
        except Exception:
            pass

    def save(self):
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "expert_profiles": [
                {
                    "expert_id": ep.expert_id,
                    "total_activations": ep.total_activations,
                    "success_activations": ep.success_activations,
                    "domain_affinities": ep.domain_affinities,
                    "time_of_day_weights": ep.time_of_day_weights,
                    "last_activated": ep.last_activated,
                    "velocity": ep.velocity,
                }
                for ep in self._expert_profiles.values()
            ],
            "domain_patterns": [
                {
                    "domain": dp.domain,
                    "hour_distribution": dp.hour_distribution,
                    "avg_k": dp.avg_k,
                    "preferred_experts": dp.preferred_experts,
                    "total_queries": dp.total_queries,
                }
                for dp in self._domain_patterns.values()
            ],
        }
        self._history_file.write_text(json.dumps(data, indent=2))

    def record_event(
        self,
        expert_ids: List[int],
        domain: str,
        k_used: int,
        confidence: float,
        mean_r_i: float,
        query_text: str,
    ):
        now = time.time()
        hour = time.localtime(now).tm_hour
        query_hash = hash(query_text) & 0xFFFFFFFF

        event = UsageEvent(
            timestamp=now, expert_ids=expert_ids, domain=domain,
            k_used=k_used, confidence=confidence, mean_r_i=mean_r_i,
            user_query_hash=query_hash,
        )
        self._events.append(event)
        self._query_domain_cache[query_hash] = domain

        is_good = confidence > 0.5 and mean_r_i > 0.2

        for eid in expert_ids:
            if eid not in self._expert_profiles:
                self._expert_profiles[eid] = ExpertProfile(expert_id=eid)
            ep = self._expert_profiles[eid]
            ep.total_activations += 1
            ep.last_activated = now
            if is_good:
                ep.success_activations += 1

            old_affinity = ep.domain_affinities.get(domain, 0.5)
            signal = mean_r_i if is_good else -0.1
            ep.domain_affinities[domain] = old_affinity + self._lr * (signal - old_affinity)

            ep.time_of_day_weights[hour] = (
                ep.time_of_day_weights[hour] * (1 - self._lr)
                + (1.0 if is_good else 0.5) * self._lr
            )

            if ep.total_activations > 1:
                success_rate = ep.success_activations / ep.total_activations
                ep.velocity = ep.velocity * self._decay_rate + success_rate * (1 - self._decay_rate)

        if domain not in self._domain_patterns:
            self._domain_patterns[domain] = DomainPattern(domain=domain)
        dp = self._domain_patterns[domain]
        dp.total_queries += 1
        dp.hour_distribution[hour] += 1.0
        dp.avg_k = dp.avg_k * 0.95 + k_used * 0.05
        self._update_preferred_experts(dp, expert_ids, is_good)

    def _update_preferred_experts(self, dp: DomainPattern, expert_ids: List[int], is_good: bool):
        if not is_good:
            return
        scores: Dict[int, float] = {}
        for eid in dp.preferred_experts:
            scores[eid] = scores.get(eid, 0) * self._decay_rate
        for eid in expert_ids:
            scores[eid] = scores.get(eid, 0) + 1.0
        dp.preferred_experts = sorted(scores, key=scores.get, reverse=True)[:20]

    def get_routing_boost(self, expert_id: int, domain: str) -> float:
        ep = self._expert_profiles.get(expert_id)
        if ep is None:
            return 0.0
        affinity = ep.domain_affinities.get(domain, 0.5)
        hour = time.localtime().tm_hour
        time_weight = ep.time_of_day_weights[hour]
        velocity = ep.velocity
        return (affinity * 0.4 + time_weight * 0.2 + velocity * 0.4) - 0.5

    def suggest_k(self, domain: str) -> int:
        dp = self._domain_patterns.get(domain)
        if dp is None:
            return configs.K_DEFAULT
        hour = time.localtime().tm_hour
        hour_weight = dp.hour_distribution[hour] / max(1, sum(dp.hour_distribution))
        base_k = dp.avg_k
        if hour_weight > 0.15:
            base_k *= 1.1
        return max(configs.K_MIN, min(configs.K_MAX, round(base_k)))

    def suggest_experts(self, domain: str, k: int) -> List[int]:
        dp = self._domain_patterns.get(domain)
        if dp is None:
            return []
        hour = time.localtime().tm_hour
        scored = []
        for eid in dp.preferred_experts:
            ep = self._expert_profiles.get(eid)
            if ep is None:
                continue
            score = (
                ep.domain_affinities.get(domain, 0.5) * 0.4
                + ep.time_of_day_weights[hour] * 0.2
                + ep.velocity * 0.4
            )
            scored.append((eid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in scored[:k]]

    def decay_all(self):
        for ep in self._expert_profiles.values():
            ep.velocity *= self._decay_rate
            for domain in ep.domain_affinities:
                ep.domain_affinities[domain] *= self._decay_rate
            ep.time_of_day_weights = [w * self._decay_rate for w in ep.time_of_day_weights]

    def get_expert_velocity_report(self, top_n: int = 10) -> List[Dict[str, Any]]:
        profiles = sorted(
            self._expert_profiles.values(),
            key=lambda p: p.velocity, reverse=True,
        )
        return [
            {
                "expert_id": p.expert_id,
                "velocity": round(p.velocity, 4),
                "activations": p.total_activations,
                "success_rate": round(p.success_activations / max(1, p.total_activations), 3),
                "top_domain": max(p.domain_affinities, key=p.domain_affinities.get) if p.domain_affinities else "none",
            }
            for p in profiles[:top_n]
        ]

    def get_user_patterns(self) -> Dict[str, Any]:
        if not self._events:
            return {"patterns": "No data yet."}
        domain_counts = defaultdict(int)
        hour_counts = [0] * 24
        for evt in self._events:
            domain_counts[evt.domain] += 1
            hour = time.localtime(evt.timestamp).tm_hour
            hour_counts[hour] += 1
        peak_hour = int(np.argmax(hour_counts))
        top_domain = max(domain_counts, key=domain_counts.get) if domain_counts else "general"
        return {
            "total_queries": len(self._events),
            "top_domain": top_domain,
            "domain_distribution": dict(domain_counts),
            "peak_hour": peak_hour,
            "session_duration_hours": (time.time() - self._session_start) / 3600,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_experts": len(self._expert_profiles),
            "tracked_domains": len(self._domain_patterns),
            "events_recorded": len(self._events),
            "user_patterns": self.get_user_patterns(),
        }

