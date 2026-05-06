from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import configs
from apex_nadir_convolution import ApexNadirConvolution
from central import CentralModel
from data import authenticate_huggingface
from experts import ExpertPool
from gating import GateModel, MaskingSchedule, TripleKSelector
from inference import InferenceEngine, InferenceResult
from memory import RoutingMemory, SessionTracker

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
class BenchmarkRecord:
    batch: int
    loop: str
    category: str
    prompt: str
    prompt_tokens: int
    output: str
    timeline: str
    loss: float
    k: int
    conf: float
    x_next: int
    thermal: float
    ram_mb: float
    ssd_read_rate_mb: float
    tok_s: float
    r_i: float
    domain: str
    experts_used: List[int]
    total_tokens: int
    latency_ms: float
    accuracy: float
    reasoning_depth: float


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
    marker_count = sum(1 for marker in reasoning_markers if marker in output.lower())
    marker_score = min(1.0, marker_count / 3.0)
    return 0.3 * length_score + 0.3 * structure_score + 0.4 * marker_score


def _build_components():
    configs.validate_config()
    authenticate_huggingface()
    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    routing_memory = RoutingMemory()
    routing_memory.load(configs.ROUTING_MEMORY_PATH)
    session_tracker = SessionTracker()
    gate = GateModel()
    gate.load()
    central = CentralModel()
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker)
    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()
    engine = InferenceEngine(
        gate=gate,
        expert_pool=expert_pool,
        central=central,
        convolution=convolution,
        routing_memory=routing_memory,
        session_tracker=session_tracker,
        triple_k=triple_k,
        masking_schedule=masking,
    )
    return engine, routing_memory


def _truncate_prompt(tokenizer, prompt: str, numerator: int, denominator: int) -> str:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return prompt
    target = max(1, int(len(token_ids) * numerator / max(denominator, 1)))
    return tokenizer.decode(token_ids[:target])


def _run_once(
    engine: InferenceEngine,
    prompt: str,
    send_to_user: bool = True,
    force_timeline_b: bool = False,
    force_timeline_a: bool = False,
):
    start = time.time()
    result = engine.run(
        prompt,
        send_to_user=send_to_user,
        force_timeline_b=force_timeline_b,
        force_timeline_a=force_timeline_a,
    )
    latency_ms = (time.time() - start) * 1000.0
    tok_s = result.token_count / max(latency_ms / 1000.0, 1e-6)
    return result, latency_ms, tok_s


def _to_record(
    batch: int,
    loop: str,
    category: str,
    prompt: str,
    expected_keywords: List[str],
    result: InferenceResult,
    latency_ms: float,
    tok_s: float,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        batch=batch,
        loop=loop,
        category=category,
        prompt=prompt,
        prompt_tokens=result.token_count,
        output=result.output_text,
        timeline=result.timeline,
        loss=0.0,
        k=result.k_used,
        conf=result.confidence,
        x_next=result.x_next,
        thermal=result.thermal_state,
        ram_mb=result.ram_headroom_mb,
        ssd_read_rate_mb=result.ssd_read_rate_mb,
        tok_s=tok_s,
        r_i=result.mean_r_i,
        domain=result.domain,
        experts_used=result.experts_activated,
        total_tokens=result.token_count,
        latency_ms=latency_ms,
        accuracy=_score_accuracy(result.output_text, expected_keywords),
        reasoning_depth=_score_reasoning_depth(result.output_text),
    )


def _append_record(path: Path, record: BenchmarkRecord) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def _print_record(record: BenchmarkRecord) -> None:
    print(
        f"batch={record.batch} | "
        f"loss={record.loss:.4f} | "
        f"k={record.k} | "
        f"conf={record.conf:.3f} | "
        f"x_next={record.x_next} | "
        f"thermal={record.thermal:.1f} | "
        f"ram_mb={record.ram_mb:.0f} | "
        f"tok/s={record.tok_s:.1f} | "
        f"r_i={record.r_i:.4f} | "
        f"domain={record.domain} | "
        f"experts_used={record.experts_used} | "
        f"total_tokens={record.total_tokens} | "
        f"loop={record.loop} | "
        f"timeline={record.timeline}"
    )


def _summarize(records: List[BenchmarkRecord]) -> Dict[str, Any]:
    by_loop: Dict[str, Dict[str, float]] = {}
    for loop_name in sorted({record.loop for record in records}):
        loop_records = [record for record in records if record.loop == loop_name]
        by_loop[loop_name] = {
            "count": len(loop_records),
            "avg_accuracy": float(np.mean([record.accuracy for record in loop_records])) if loop_records else 0.0,
            "avg_reasoning_depth": float(np.mean([record.reasoning_depth for record in loop_records])) if loop_records else 0.0,
            "avg_latency_ms": float(np.mean([record.latency_ms for record in loop_records])) if loop_records else 0.0,
            "avg_tok_s": float(np.mean([record.tok_s for record in loop_records])) if loop_records else 0.0,
            "avg_k": float(np.mean([record.k for record in loop_records])) if loop_records else 0.0,
            "avg_conf": float(np.mean([record.conf for record in loop_records])) if loop_records else 0.0,
            "avg_r_i": float(np.mean([record.r_i for record in loop_records])) if loop_records else 0.0,
            "avg_x_next": float(np.mean([record.x_next for record in loop_records])) if loop_records else 0.0,
        }
    return {"loops": by_loop, "records": len(records)}


def run_benchmark(output_dir: str = "logs", clear_existing: bool = True) -> Dict[str, Any]:
    engine, routing_memory = _build_components()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "benchmark_runs.jsonl"
    summary_path = output_root / "benchmark_summary.json"
    if clear_existing:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    tokenizer = engine.gate.tokenizer
    batch = 0
    records: List[BenchmarkRecord] = []
    for item in BENCHMARK_PROMPTS:
        prompt = item["prompt"]
        category = item["category"]
        expected_keywords = item["expected_keywords"]
        half_prompt = _truncate_prompt(tokenizer, prompt, 1, 2)
        centile_prompt = _truncate_prompt(tokenizer, prompt, 1, 100)

        batch += 1
        result_b_full, latency_b_full, tok_s_b_full = _run_once(engine, prompt, send_to_user=True, force_timeline_b=True)
        record_b_full = _to_record(batch, "training_b_full", category, prompt, expected_keywords, result_b_full, latency_b_full, tok_s_b_full)
        records.append(record_b_full)
        _append_record(records_path, record_b_full)
        _print_record(record_b_full)

        batch += 1
        result_deploy, latency_deploy, tok_s_deploy = _run_once(engine, half_prompt, send_to_user=True)
        record_deploy = _to_record(batch, "deployment_half", category, half_prompt, expected_keywords, result_deploy, latency_deploy, tok_s_deploy)
        records.append(record_deploy)
        _append_record(records_path, record_deploy)
        _print_record(record_deploy)

        if result_deploy.timeline == "A":
            batch += 1
            shadow_result, shadow_latency, shadow_tok_s = _run_once(engine, half_prompt, send_to_user=False, force_timeline_b=True)
            shadow_record = _to_record(batch, "deployment_half_shadow_b", category, half_prompt, expected_keywords, shadow_result, shadow_latency, shadow_tok_s)
            records.append(shadow_record)
            _append_record(records_path, shadow_record)
            _print_record(shadow_record)

        batch += 1
        result_a_only, latency_a_only, tok_s_a_only = _run_once(engine, centile_prompt, send_to_user=True, force_timeline_a=True)
        record_a_only = _to_record(batch, "timeline_a_centile", category, centile_prompt, expected_keywords, result_a_only, latency_a_only, tok_s_a_only)
        records.append(record_a_only)
        _append_record(records_path, record_a_only)
        _print_record(record_a_only)

    summary = _summarize(records)
    summary["routing_clusters"] = len(routing_memory.clusters)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"saved_records={records_path}")
    print(f"saved_summary={summary_path}")
    return summary


if __name__ == "__main__":
    run_benchmark()
