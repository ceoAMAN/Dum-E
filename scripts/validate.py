from __future__ import annotations
import json
import shutil
import sys
import time
from pathlib import Path
import argparse
from typing import Dict, List, Any
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import mlx.core as mx
import configs
from central import CentralModel
from experts import ExpertPool
from gating import GateModel, TripleKSelector, MaskingSchedule
from memory import RoutingMemory, SessionTracker
from apex_nadir_convolution import ApexNadirConvolution
from inference import InferenceEngine
from data import authenticate_huggingface

SAMPLE_PROMPTS: List[str] = [
    "hello world",
    "explain eigenvectors in hilbert space",
    "write a python function to reverse a list",
    "summarize the major steps in a data pipeline",
    "what is photosynthesis",
    "design a cache eviction strategy",
    "prove that the sum of two even numbers is even",
    "generate a JSON schema for a user profile",
    "how to fix a memory leak in python",
    "describe transformer attention",
]

def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")

def _write_json(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(record, handle, indent=2)

def _update_benchmark_total_validation(summary: Dict[str, Any]) -> None:
    total_path = configs.LOG_DIR / "benchmark_total_validation.json"
    if not total_path.exists():
        return
    try:
        with total_path.open("r") as handle:
            total = json.load(handle)
    except Exception:
        return
    total["runtime_validation"] = summary
    validation_runs_path = configs.LOG_DIR / "validation_runs.jsonl"
    benchmark_validation_runs_path = configs.LOG_DIR / "benchmark_validation_runs.jsonl"
    if validation_runs_path.exists():
        shutil.copyfile(validation_runs_path, benchmark_validation_runs_path)
    total.setdefault("training_validation", {})["validation_summary"] = {
        "exists": True,
        "source_path": str(configs.LOG_DIR / "validation_summary.json"),
        "benchmark_path": str(configs.LOG_DIR / "benchmark_validation_summary.json"),
        "records": 1,
        "latest": summary,
    }
    total.setdefault("training_validation", {})["validation_runs"] = {
        "exists": validation_runs_path.exists(),
        "source_path": str(validation_runs_path),
        "benchmark_path": str(benchmark_validation_runs_path),
    }
    _write_json(total_path, total)
    _write_json(configs.LOG_DIR / "benchmark_validation_summary.json", summary)

def run(samples: int = 50) -> Dict[str, Any]:
    print("[validate] Running validation")
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
        gate=gate, expert_pool=expert_pool, central=central,
        convolution=convolution, routing_memory=routing_memory,
        session_tracker=session_tracker, triple_k=triple_k,
        masking_schedule=masking,
    )
    ks: List[int] = []
    fast_path = 0
    timeline_b = 0
    validation_runs_path = configs.LOG_DIR / "validation_runs.jsonl"
    validation_runs_path.unlink(missing_ok=True)
    print("[validate] Phase 1: routing distribution")
    for i in range(samples):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        token_ids = gate.tokenizer.encode(prompt)
        tokens = mx.array(token_ids)
        gate_out = gate.forward(tokens)
        if gate_out.confidence > configs.FAST_PATH_THRESHOLD:
            fast_path += 1
            ks.append(0)
            predicted_timeline = "A"
        else:
            timeline_b += 1
            ks.append(gate_out.k_per_token)
            predicted_timeline = "B"
        _append_jsonl(
            validation_runs_path,
            {
                "record_type": "routing_distribution",
                "sample": i + 1,
                "prompt": prompt,
                "tokens": len(token_ids),
                "confidence": float(gate_out.confidence),
                "k": int(ks[-1]),
                "predicted_timeline": predicted_timeline,
            },
        )
    k_arr = np.array(ks)
    print(f"[validate] Fast-path rate: {fast_path / samples:.2f}")
    print(f"[validate] Timeline-B rate: {timeline_b / samples:.2f}")
    print(f"[validate] K mean/min/max: {k_arr.mean():.2f}/{k_arr.min()}/{k_arr.max()}")
    print("[validate] Phase 2: end-to-end execution")
    test_prompts = SAMPLE_PROMPTS[:3]
    execution_records: List[Dict[str, Any]] = []
    for prompt in test_prompts:
        start = time.time()
        result = engine.run(prompt, send_to_user=True)
        latency_ms = (time.time() - start) * 1000
        execution_record = {
            "record_type": "end_to_end_execution",
            "prompt": prompt,
            "k": int(result.k_used),
            "timeline": result.timeline,
            "experts": result.experts_activated,
            "latency_ms": float(latency_ms),
            "confidence": float(result.confidence),
            "domain": result.domain,
            "r_i": float(result.mean_r_i),
            "x_next": int(result.x_next),
            "thermal": float(result.thermal_state),
        }
        execution_records.append(execution_record)
        _append_jsonl(validation_runs_path, execution_record)
        print(
            f"[validate] '{prompt[:30]}...' -> "
            f"K={result.k_used} timeline={result.timeline} "
            f"experts={result.experts_activated} latency={latency_ms:.0f}ms"
        )
    print("[validate] Phase 3: cluster & session stats")
    print(f"[validate] Routing clusters: {len(routing_memory.clusters)}")
    print(f"[validate] Session tokens: {session_tracker.get_total_tokens_seen()}")
    print(f"[validate] Timeline-A rate: {session_tracker.get_timeline_a_rate():.3f}")
    summary = {
        "record_type": "runtime_validation_summary",
        "samples": samples,
        "fast_path_rate": float(fast_path / samples),
        "timeline_b_rate": float(timeline_b / samples),
        "k_mean": float(k_arr.mean()),
        "k_min": int(k_arr.min()),
        "k_max": int(k_arr.max()),
        "execution_records": execution_records,
        "routing_clusters": len(routing_memory.clusters),
        "session_tokens": session_tracker.get_total_tokens_seen(),
        "timeline_a_rate": float(session_tracker.get_timeline_a_rate()),
        "runs_path": str(validation_runs_path),
    }
    _write_json(configs.LOG_DIR / "validation_summary.json", summary)
    _update_benchmark_total_validation(summary)
    print("[validate] DONE")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    run(samples=args.samples)
