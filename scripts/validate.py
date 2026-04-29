from __future__ import annotations
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

def run(samples: int = 50) -> None:
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
    print("[validate] Phase 1: routing distribution")
    for i in range(samples):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        token_ids = gate.tokenizer.encode(prompt)
        tokens = mx.array(token_ids)
        gate_out = gate.forward(tokens)
        if gate_out.confidence > configs.FAST_PATH_THRESHOLD:
            fast_path += 1
            ks.append(0)
        else:
            timeline_b += 1
            ks.append(gate_out.k_per_token)
    k_arr = np.array(ks)
    print(f"[validate] Fast-path rate: {fast_path / samples:.2f}")
    print(f"[validate] Timeline-B rate: {timeline_b / samples:.2f}")
    print(f"[validate] K mean/min/max: {k_arr.mean():.2f}/{k_arr.min()}/{k_arr.max()}")
    print("[validate] Phase 2: end-to-end execution")
    test_prompts = SAMPLE_PROMPTS[:3]
    for prompt in test_prompts:
        start = time.time()
        result = engine.run(prompt, send_to_user=True)
        latency_ms = (time.time() - start) * 1000
        print(
            f"[validate] '{prompt[:30]}...' -> "
            f"K={result.k_used} timeline={result.timeline} "
            f"experts={result.experts_activated} latency={latency_ms:.0f}ms"
        )
    print("[validate] Phase 3: cluster & session stats")
    print(f"[validate] Routing clusters: {len(routing_memory.clusters)}")
    print(f"[validate] Session tokens: {session_tracker.get_total_tokens_seen()}")
    print(f"[validate] Timeline-A rate: {session_tracker.get_timeline_a_rate():.3f}")
    print("[validate] DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    run(samples=args.samples)
