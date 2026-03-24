# pyre-unsafe
"""Routing memory and reward-based expert scoring."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import configs


@dataclass
class RoutingRecord:
    prompt_hash: str
    expert_indices: List[int]
    quality_score: float
    diversity_score: float
    timestamp: float
    k: int
    timeline: str


class ExpertRewardTracker:
    def __init__(self, num_experts: int = configs.NUM_EXPERTS, decay: float = 0.95) -> None:
        self.num_experts = num_experts
        self.decay = decay
        self.reward_scores = np.zeros(num_experts, dtype=np.float64)
        self.selection_count = np.zeros(num_experts, dtype=np.int64)
        self.pair_rewards: Dict[Tuple[int, int], float] = defaultdict(float)
        self.pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    def update(self, expert_indices: List[int], quality: float, diversity: float) -> None:
        reward = 0.6 * quality + 0.4 * diversity
        for idx in expert_indices:
            if 0 <= idx < self.num_experts:
                self.reward_scores[idx] = self.decay * self.reward_scores[idx] + (1 - self.decay) * reward
                self.selection_count[idx] += 1

        for i in range(len(expert_indices)):
            for j in range(i + 1, len(expert_indices)):
                a, b = min(expert_indices[i], expert_indices[j]), max(expert_indices[i], expert_indices[j])
                pair = (a, b)
                self.pair_rewards[pair] = self.decay * self.pair_rewards[pair] + (1 - self.decay) * reward
                self.pair_counts[pair] += 1

    def get_expert_scores(self) -> np.ndarray:
        return self.reward_scores.copy()

    def get_bonus_for_combination(self, expert_indices: List[int]) -> float:
        if len(expert_indices) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(expert_indices)):
            for j in range(i + 1, len(expert_indices)):
                a, b = min(expert_indices[i], expert_indices[j]), max(expert_indices[i], expert_indices[j])
                pair = (a, b)
                if self.pair_counts[pair] > 0:
                    total += self.pair_rewards[pair]
                    count += 1
        return total / max(count, 1)

    def top_experts(self, k: int) -> List[int]:
        return list(np.argsort(self.reward_scores)[-k:][::-1])

    def top_pairs(self, n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        sorted_pairs = sorted(self.pair_rewards.items(), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:n]


class RoutingMemory:
    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self.records: Dict[str, List[RoutingRecord]] = defaultdict(list)
        self.reward_tracker = ExpertRewardTracker()

    def _prompt_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _prompt_category(self, text: str) -> str:
        text_lower = text.lower()
        if any(w in text_lower for w in ["code", "function", "program", "debug", "python", "implement"]):
            return "code"
        if any(w in text_lower for w in ["prove", "theorem", "logic", "reason", "why", "explain how"]):
            return "reasoning"
        if any(w in text_lower for w in ["summarize", "describe", "what is", "tell me"]):
            return "knowledge"
        return "general"

    def store(
        self,
        prompt: str,
        expert_indices: List[int],
        quality: float,
        diversity: float,
        k: int,
        timeline: str,
    ) -> None:
        ph = self._prompt_hash(prompt)
        record = RoutingRecord(
            prompt_hash=ph,
            expert_indices=expert_indices,
            quality_score=quality,
            diversity_score=diversity,
            timestamp=time.time(),
            k=k,
            timeline=timeline,
        )
        self.records[ph].append(record)
        if len(self.records[ph]) > 20:
            self.records[ph] = sorted(self.records[ph], key=lambda r: r.quality_score, reverse=True)[:10]
        self.reward_tracker.update(expert_indices, quality, diversity)

        total = sum(len(v) for v in self.records.values())
        if total > self.max_size:
            self._evict()

    def _evict(self) -> None:
        all_records = []
        for ph, recs in self.records.items():
            for r in recs:
                all_records.append((ph, r))
        all_records.sort(key=lambda x: x[1].timestamp)
        remove_count = len(all_records) - self.max_size
        for i in range(remove_count):
            ph, rec = all_records[i]
            self.records[ph].remove(rec)
            if not self.records[ph]:
                del self.records[ph]

    def lookup(self, prompt: str) -> Optional[List[int]]:
        ph = self._prompt_hash(prompt)
        recs = self.records.get(ph)
        if recs:
            best = max(recs, key=lambda r: r.quality_score)
            return best.expert_indices
        return None

    def suggest_experts(self, prompt: str, k: int) -> Optional[List[int]]:
        exact = self.lookup(prompt)
        if exact:
            return exact[:k]

        category = self._prompt_category(prompt)
        category_records: List[RoutingRecord] = []
        for recs in self.records.values():
            category_records.extend(recs)

        if not category_records:
            return None

        expert_quality: Dict[int, List[float]] = defaultdict(list)
        for r in category_records:
            for idx in r.expert_indices:
                expert_quality[idx].append(r.quality_score)

        avg_quality = {idx: np.mean(scores) for idx, scores in expert_quality.items()}
        sorted_experts = sorted(avg_quality.keys(), key=lambda x: avg_quality[x], reverse=True)

        reward_scores = self.reward_tracker.get_expert_scores()
        combined = []
        for idx in sorted_experts[:k * 2]:
            score = 0.5 * avg_quality.get(idx, 0.0) + 0.5 * float(reward_scores[idx])
            combined.append((idx, score))
        combined.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in combined[:k]]

    def get_stats(self) -> Dict[str, Any]:
        total_records = sum(len(v) for v in self.records.values())
        unique_prompts = len(self.records)
        top_experts = self.reward_tracker.top_experts(10)
        top_pairs = self.reward_tracker.top_pairs(5)
        scores = self.reward_tracker.get_expert_scores()
        return {
            "total_records": total_records,
            "unique_prompts": unique_prompts,
            "top_experts": top_experts,
            "top_expert_scores": [float(scores[i]) for i in top_experts],
            "top_pairs": [(p, float(s)) for p, s in top_pairs],
            "mean_reward": float(scores.mean()),
            "max_reward": float(scores.max()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": {},
            "reward_scores": self.reward_tracker.reward_scores.tolist(),
            "selection_count": self.reward_tracker.selection_count.tolist(),
        }
        for ph, recs in self.records.items():
            data["records"][ph] = [
                {
                    "expert_indices": r.expert_indices,
                    "quality_score": r.quality_score,
                    "diversity_score": r.diversity_score,
                    "timestamp": r.timestamp,
                    "k": r.k,
                    "timeline": r.timeline,
                }
                for r in recs
            ]
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self.reward_tracker.reward_scores = np.array(data["reward_scores"], dtype=np.float64)
            self.reward_tracker.selection_count = np.array(data["selection_count"], dtype=np.int64)
            for ph, recs in data["records"].items():
                self.records[ph] = [
                    RoutingRecord(
                        prompt_hash=ph,
                        expert_indices=r["expert_indices"],
                        quality_score=r["quality_score"],
                        diversity_score=r["diversity_score"],
                        timestamp=r["timestamp"],
                        k=r["k"],
                        timeline=r["timeline"],
                    )
                    for r in recs
                ]
            return True
        except Exception:
            return False
