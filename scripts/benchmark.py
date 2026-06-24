from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

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

TRAINING_VALIDATION_ARTIFACTS = {
    "k_trajectory": ("k_trajectory.jsonl", "benchmark_k_trajectory.jsonl", "jsonl"),
    "expert_drift": ("expert_drift.jsonl", "benchmark_expert_drift.jsonl", "jsonl"),
    "thermal_regression": (
        "thermal_regression_validation.jsonl",
        "benchmark_thermal_regression_validation.jsonl",
        "jsonl",
    ),
    "proof_metrics": ("proof_metrics.jsonl", "benchmark_proof_metrics.jsonl", "jsonl"),
    "finetune_metrics": ("finetune_metrics.json", "benchmark_finetune_metrics.json", "json"),
    "validation_summary": ("validation_summary.json", "benchmark_validation_summary.json", "json"),
    "validation_runs": ("validation_runs.jsonl", "benchmark_validation_runs.jsonl", "jsonl"),
    "fresh_training_log": ("sturnus-fresh-10m.log", "benchmark_training.log", "text"),
    "full_protocol_log": ("sturnus-full-protocol.log", "benchmark_full_protocol.log", "text"),
}


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
    min_experts: int = 0,
):
    start = time.time()
    result = engine.run(
        prompt,
        send_to_user=send_to_user,
        force_timeline_b=force_timeline_b,
        force_timeline_a=force_timeline_a,
        min_experts=min_experts,
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


def _jsonl_stats(path: Path, tail_count: int = 5) -> Dict[str, Any]:
    records = 0
    tail: List[Any] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                parsed: Any = json.loads(line)
            except json.JSONDecodeError:
                parsed = {"raw": line}
            tail.append(parsed)
            if len(tail) > tail_count:
                tail.pop(0)
    return {
        "records": records,
        "latest": tail[-1] if tail else None,
        "tail": tail,
    }


def _json_stats(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        data = json.load(handle)
    return {"records": 1, "latest": data}


def _text_stats(path: Path, tail_count: int = 20) -> Dict[str, Any]:
    tail: List[str] = []
    lines = 0
    with path.open("r", errors="replace") as handle:
        for line in handle:
            lines += 1
            tail.append(line.rstrip("\n"))
            if len(tail) > tail_count:
                tail.pop(0)
    return {"lines": lines, "tail": tail}


def _snapshot_training_validation(output_root: Path) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    for name, (source_name, dest_name, artifact_type) in TRAINING_VALIDATION_ARTIFACTS.items():
        source = output_root / source_name
        dest = output_root / dest_name
        if not source.exists():
            dest.unlink(missing_ok=True)
            artifacts[name] = {
                "exists": False,
                "source_path": str(source),
                "benchmark_path": str(dest),
            }
            continue
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
        if artifact_type == "jsonl":
            stats = _jsonl_stats(dest)
        elif artifact_type == "json":
            stats = _json_stats(dest)
        else:
            stats = _text_stats(dest)
        artifacts[name] = {
            "exists": True,
            "source_path": str(source),
            "benchmark_path": str(dest),
            "bytes": dest.stat().st_size,
            **stats,
        }
    return artifacts


def _write_total_validation(
    total_validation_path: Path,
    records: List[BenchmarkRecord],
    summary: Dict[str, Any],
    training_validation: Dict[str, Any],
    records_path: Path,
    summary_path: Path,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    total_validation = {
        "record_type": "benchmark_total_validation",
        "saved_at_unix": time.time(),
        "benchmark": {
            "records": len(records),
            "last_batch": records[-1].batch if records else 0,
            "records_path": str(records_path),
            "summary_path": str(summary_path),
            "checkpoint_path": str(checkpoint_path),
            "summary": summary,
        },
        "training_validation": training_validation,
    }
    with total_validation_path.open("w") as handle:
        json.dump(total_validation, handle, indent=2)
    return total_validation


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


def _write_summary(
    summary_path: Path,
    records: List[BenchmarkRecord],
    routing_memory: RoutingMemory,
) -> Dict[str, Any]:
    summary = _summarize(records)
    summary["routing_clusters"] = len(routing_memory.clusters)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _save_checkpoint(
    checkpoint_path: Path,
    records: List[BenchmarkRecord],
    routing_memory: RoutingMemory,
    summary_path: Path,
    records_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    summary = _write_summary(summary_path, records, routing_memory)
    total_validation_path = output_root / "benchmark_total_validation.json"
    training_validation = _snapshot_training_validation(output_root)
    total_validation = _write_total_validation(
        total_validation_path,
        records,
        summary,
        training_validation,
        records_path,
        summary_path,
        checkpoint_path,
    )
    summary["total_validation_path"] = str(total_validation_path)
    summary["training_validation"] = training_validation
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    checkpoint = {
        "saved_at_unix": time.time(),
        "records": len(records),
        "last_batch": records[-1].batch if records else 0,
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "total_validation_path": str(total_validation_path),
        "training_validation": total_validation["training_validation"],
        "summary": summary,
    }
    with checkpoint_path.open("w") as handle:
        json.dump(checkpoint, handle, indent=2)
    print(f"saved_checkpoint={checkpoint_path} | records={len(records)}")
    return summary


def _maybe_save_checkpoint(
    checkpoint_path: Path,
    records: List[BenchmarkRecord],
    routing_memory: RoutingMemory,
    summary_path: Path,
    records_path: Path,
    output_root: Path,
    save_every_runs: int,
) -> None:
    if save_every_runs <= 0:
        return
    if records and len(records) % save_every_runs == 0:
        _save_checkpoint(checkpoint_path, records, routing_memory, summary_path, records_path, output_root)


def run_benchmark(output_dir: str = "logs", clear_existing: bool = True, save_every_runs: int = 100) -> Dict[str, Any]:
    engine, routing_memory = _build_components()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "benchmark_runs.jsonl"
    summary_path = output_root / "benchmark_summary.json"
    checkpoint_path = output_root / "benchmark_checkpoint.json"
    if clear_existing:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
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
        result_b_full, latency_b_full, tok_s_b_full = _run_once(
            engine,
            prompt,
            send_to_user=True,
            force_timeline_b=True,
            min_experts=max(1, routing_memory.get_domain_mean_k()),
        )
        record_b_full = _to_record(batch, "training_b_full", category, prompt, expected_keywords, result_b_full, latency_b_full, tok_s_b_full)
        records.append(record_b_full)
        _append_record(records_path, record_b_full)
        _print_record(record_b_full)
        _maybe_save_checkpoint(
            checkpoint_path,
            records,
            routing_memory,
            summary_path,
            records_path,
            output_root,
            save_every_runs,
        )

        batch += 1
        result_deploy, latency_deploy, tok_s_deploy = _run_once(engine, half_prompt, send_to_user=True)
        record_deploy = _to_record(batch, "deployment_half", category, half_prompt, expected_keywords, result_deploy, latency_deploy, tok_s_deploy)
        records.append(record_deploy)
        _append_record(records_path, record_deploy)
        _print_record(record_deploy)
        _maybe_save_checkpoint(
            checkpoint_path,
            records,
            routing_memory,
            summary_path,
            records_path,
            output_root,
            save_every_runs,
        )

        if result_deploy.timeline == "A":
            batch += 1
            shadow_result, shadow_latency, shadow_tok_s = _run_once(
                engine,
                half_prompt,
                send_to_user=False,
                force_timeline_b=True,
                min_experts=max(1, routing_memory.get_domain_mean_k()),
            )
            shadow_record = _to_record(batch, "deployment_half_shadow_b", category, half_prompt, expected_keywords, shadow_result, shadow_latency, shadow_tok_s)
            records.append(shadow_record)
            _append_record(records_path, shadow_record)
            _print_record(shadow_record)
            _maybe_save_checkpoint(
                checkpoint_path,
                records,
                routing_memory,
                summary_path,
                records_path,
                output_root,
                save_every_runs,
            )

        batch += 1
        result_a_only, latency_a_only, tok_s_a_only = _run_once(engine, centile_prompt, send_to_user=True, force_timeline_a=True)
        record_a_only = _to_record(batch, "timeline_a_centile", category, centile_prompt, expected_keywords, result_a_only, latency_a_only, tok_s_a_only)
        records.append(record_a_only)
        _append_record(records_path, record_a_only)
        _print_record(record_a_only)
        _maybe_save_checkpoint(
            checkpoint_path,
            records,
            routing_memory,
            summary_path,
            records_path,
            output_root,
            save_every_runs,
        )

    summary = _save_checkpoint(checkpoint_path, records, routing_memory, summary_path, records_path, output_root)
    print(f"saved_records={records_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_total_validation={output_root / 'benchmark_total_validation.json'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sturnus benchmark loops")
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--save-every-runs", type=int, default=100)
    args = parser.parse_args()
    run_benchmark(
        output_dir=args.output_dir,
        save_every_runs=args.save_every_runs,
    )
