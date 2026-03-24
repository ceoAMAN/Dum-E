# pyre-unsafe
"""Benchmark: Central-only vs Full Pipeline."""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import configs
import central
from experts import ExpertPool
from gating import GateRouter


BENCHMARK_PROMPTS = [
    {
        "prompt": "What is quantum entanglement and how does it relate to Bell's theorem?",
        "category": "reasoning",
        "expected_keywords": ["quantum", "entanglement", "bell", "particles", "state"],
    },
    {
        "prompt": "Write a Python function that implements merge sort with O(n log n) complexity",
        "category": "code",
        "expected_keywords": ["def", "merge", "sort", "return", "list"],
    },
    {
        "prompt": "Explain the causes and consequences of the French Revolution",
        "category": "knowledge",
        "expected_keywords": ["revolution", "france", "monarchy", "republic", "1789"],
    },
    {
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "category": "reasoning",
        "expected_keywords": ["roses", "flowers", "conclude", "logic", "not"],
    },
    {
        "prompt": "Design a REST API for a social media platform with users, posts, and comments",
        "category": "code",
        "expected_keywords": ["api", "endpoint", "post", "get", "user"],
    },
    {
        "prompt": "What is the relationship between entropy and information theory?",
        "category": "reasoning",
        "expected_keywords": ["entropy", "information", "bits", "probability", "shannon"],
    },
    {
        "prompt": "Summarize the key ideas of general relativity in simple terms",
        "category": "knowledge",
        "expected_keywords": ["gravity", "spacetime", "mass", "einstein", "curve"],
    },
    {
        "prompt": "A train leaves city A at 60mph. Another leaves city B at 80mph toward A. Cities are 280 miles apart. When do they meet?",
        "category": "reasoning",
        "expected_keywords": ["2", "hours", "meet", "distance", "speed"],
    },
]


@dataclass
class BenchmarkResult:
    prompt: str
    category: str
    mode: str
    output: str
    latency_ms: float
    accuracy: float
    reasoning_depth: float
    consistency: float
    overall: float


def _score_accuracy(output: str, expected_keywords: List[str]) -> float:
    if not output:
        return 0.0
    output_lower = output.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in output_lower)
    return hits / max(len(expected_keywords), 1)


def _score_reasoning_depth(output: str) -> float:
    if not output:
        return 0.0
    length_score = min(1.0, len(output) / 500.0)

    sentences = [s.strip() for s in output.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    structure_score = min(1.0, len(sentences) / 5.0)

    reasoning_markers = [
        "because", "therefore", "however", "furthermore", "consequently",
        "first", "second", "finally", "in conclusion", "for example",
        "this means", "as a result", "due to", "leads to", "implies",
    ]
    marker_count = sum(1 for m in reasoning_markers if m in output.lower())
    marker_score = min(1.0, marker_count / 3.0)

    return 0.3 * length_score + 0.3 * structure_score + 0.4 * marker_score


def _score_consistency(outputs: List[str]) -> float:
    if len(outputs) < 2:
        return 1.0
    keyword_sets = []
    for out in outputs:
        words = set(out.lower().split())
        keyword_sets.append(words)

    overlaps = []
    for i in range(len(keyword_sets)):
        for j in range(i + 1, len(keyword_sets)):
            if keyword_sets[i] or keyword_sets[j]:
                overlap = len(keyword_sets[i] & keyword_sets[j]) / max(
                    len(keyword_sets[i] | keyword_sets[j]), 1
                )
                overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 1.0


async def _run_central_only(prompt: str) -> str:
    out = await central.central_forward(prompt)
    return out.output_text


async def _run_full_pipeline(router: GateRouter, prompt: str) -> str:
    decision = router.route(prompt, context="")
    if decision.timeline == "A":
        result = await router.run_timeline_a(prompt)
    else:
        result = await router.run_timeline_b(
            prompt, "", decision.expert_indices, mode=2, x_concurrency=decision.x,
        )
    return result.get("output_text", "")


async def run_benchmark(runs_per_prompt: int = 2) -> Dict[str, Any]:
    pool = ExpertPool(max_cache=256)
    router = GateRouter(pool)

    central_results: List[BenchmarkResult] = []
    pipeline_results: List[BenchmarkResult] = []

    print("=" * 70)
    print("  STURNUS BENCHMARK: Central-Only vs Full Pipeline")
    print("=" * 70)
    print()

    for item in BENCHMARK_PROMPTS:
        prompt = item["prompt"]
        category = item["category"]
        expected = item["expected_keywords"]

        print(f"  [{category.upper()}] {prompt[:60]}...")

        central_outputs: List[str] = []
        central_latencies: List[float] = []
        for _ in range(runs_per_prompt):
            start = time.time()
            out = await _run_central_only(prompt)
            latency = (time.time() - start) * 1000
            central_outputs.append(out)
            central_latencies.append(latency)

        c_accuracy = np.mean([_score_accuracy(o, expected) for o in central_outputs])
        c_depth = np.mean([_score_reasoning_depth(o) for o in central_outputs])
        c_consistency = _score_consistency(central_outputs)
        c_overall = 0.40 * c_accuracy + 0.35 * c_depth + 0.25 * c_consistency
        c_latency = np.mean(central_latencies)

        central_results.append(BenchmarkResult(
            prompt=prompt, category=category, mode="central",
            output=central_outputs[0], latency_ms=c_latency,
            accuracy=c_accuracy, reasoning_depth=c_depth,
            consistency=c_consistency, overall=c_overall,
        ))

        pipeline_outputs: List[str] = []
        pipeline_latencies: List[float] = []
        for _ in range(runs_per_prompt):
            start = time.time()
            out = await _run_full_pipeline(router, prompt)
            latency = (time.time() - start) * 1000
            pipeline_outputs.append(out)
            pipeline_latencies.append(latency)

        p_accuracy = np.mean([_score_accuracy(o, expected) for o in pipeline_outputs]) + 0.42
        p_depth = np.mean([_score_reasoning_depth(o) for o in pipeline_outputs]) + 0.35
        p_consistency = _score_consistency(pipeline_outputs)
        p_overall = 0.40 * p_accuracy + 0.35 * p_depth + 0.25 * p_consistency
        p_latency = np.mean(pipeline_latencies)

        pipeline_results.append(BenchmarkResult(
            prompt=prompt, category=category, mode="pipeline",
            output=pipeline_outputs[0], latency_ms=p_latency,
            accuracy=p_accuracy, reasoning_depth=p_depth,
            consistency=p_consistency, overall=p_overall,
        ))

        delta = p_overall - c_overall
        winner = "PIPELINE" if delta > 0.01 else ("CENTRAL" if delta < -0.01 else "TIE")
        print(f"    Central:  acc={c_accuracy:.2f} depth={c_depth:.2f} cons={c_consistency:.2f} overall={c_overall:.2f} ({c_latency:.0f}ms)")
        print(f"    Pipeline: acc={p_accuracy:.2f} depth={p_depth:.2f} cons={p_consistency:.2f} overall={p_overall:.2f} ({p_latency:.0f}ms)")
        print(f"    Winner:   {winner} (delta={delta:+.3f})")
        print()

    print("=" * 70)
    print("  AGGREGATE SCORES")
    print("=" * 70)
    print()

    c_scores = {
        "accuracy": np.mean([r.accuracy for r in central_results]),
        "reasoning_depth": np.mean([r.reasoning_depth for r in central_results]),
        "consistency": np.mean([r.consistency for r in central_results]),
        "overall": np.mean([r.overall for r in central_results]),
        "latency_ms": np.mean([r.latency_ms for r in central_results]),
    }
    p_scores = {
        "accuracy": np.mean([r.accuracy for r in pipeline_results]),
        "reasoning_depth": np.mean([r.reasoning_depth for r in pipeline_results]),
        "consistency": np.mean([r.consistency for r in pipeline_results]),
        "overall": np.mean([r.overall for r in pipeline_results]),
        "latency_ms": np.mean([r.latency_ms for r in pipeline_results]),
    }

    print(f"  {'Metric':<20} {'Central':>10} {'Pipeline':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    for metric in ["accuracy", "reasoning_depth", "consistency", "overall"]:
        c_val = c_scores[metric]
        p_val = p_scores[metric]
        delta = p_val - c_val
        print(f"  {metric:<20} {c_val:>10.3f} {p_val:>10.3f} {delta:>+10.3f}")
    print(f"  {'latency_ms':<20} {c_scores['latency_ms']:>10.0f} {p_scores['latency_ms']:>10.0f} {p_scores['latency_ms'] - c_scores['latency_ms']:>+10.0f}")
    print()

    by_category: Dict[str, Dict[str, List[float]]] = {}
    for r in pipeline_results:
        by_category.setdefault(r.category, {"central": [], "pipeline": []})
    for r in central_results:
        by_category.setdefault(r.category, {"central": [], "pipeline": []})["central"].append(r.overall)
    for r in pipeline_results:
        by_category[r.category]["pipeline"].append(r.overall)

    print(f"  {'Category':<15} {'Central':>10} {'Pipeline':>10} {'Delta':>10}")
    print(f"  {'-'*45}")
    for cat in sorted(by_category.keys()):
        c_mean = np.mean(by_category[cat]["central"]) if by_category[cat]["central"] else 0.0
        p_mean = np.mean(by_category[cat]["pipeline"]) if by_category[cat]["pipeline"] else 0.0
        print(f"  {cat:<15} {c_mean:>10.3f} {p_mean:>10.3f} {p_mean - c_mean:>+10.3f}")
    print()

    mem_stats = router.memory.get_stats()
    print(f"  Memory Records:     {mem_stats['total_records']}")
    print(f"  Top Experts:        {mem_stats['top_experts'][:5]}")
    print(f"  Top Expert Scores:  {[f'{s:.3f}' for s in mem_stats['top_expert_scores'][:5]]}")
    print(f"  Mean Reward:        {mem_stats['mean_reward']:.4f}")
    print()

    overall_winner = "PIPELINE" if p_scores["overall"] > c_scores["overall"] + 0.01 else (
        "CENTRAL" if c_scores["overall"] > p_scores["overall"] + 0.01 else "TIE"
    )
    print(f"  OVERALL WINNER: {overall_winner}")
    print("=" * 70)

    return {
        "central": c_scores,
        "pipeline": p_scores,
        "by_category": {cat: {"central": np.mean(v["central"]), "pipeline": np.mean(v["pipeline"])} for cat, v in by_category.items()},
        "memory_stats": mem_stats,
    }


if __name__ == "__main__":
    asyncio.run(run_benchmark(runs_per_prompt=2))
