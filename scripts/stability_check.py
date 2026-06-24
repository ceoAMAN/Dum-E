#!/usr/bin/env python
from __future__ import annotations
import sys
import time
import asyncio
import argparse
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from main import boot_system, DeadTimeState, dead_time_orchestrator, process_input, run_pending_timeline_a_shadows

SAMPLE_PROMPTS = [
    "What is quantum entanglement and how does it relate to Bell's theorem?",
    "Write a Python function that implements merge sort with O(n log n) complexity",
    "Explain the causes and consequences of the French Revolution",
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

async def run_stability_test(duration_secs: int, query_interval: int):
    print("=" * 80)
    print(f"  STURNUS STABILITY CHECK (Duration: {duration_secs}s, Interval: {query_interval}s)")
    print("=" * 80)
    
    # Force deployment mode so we are validating the deployment behavior
    configs.DEPLOYMENT = True
    
    print("Booting Sturnus system...")
    components = boot_system()
    dead_state = DeadTimeState()
    
    # Start dead-time B shadow task
    orchestrator_task = asyncio.create_task(dead_time_orchestrator(components, dead_state))
    
    start_time = time.time()
    end_time = start_time + duration_secs
    
    log_records = []
    log_file = ROOT / "logs" / "stability_metrics.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear previous log
    if log_file.exists():
        log_file.unlink()
        
    query_count = 0
    try:
        while time.time() < end_time:
            elapsed = time.time() - start_time
            query = random.choice(SAMPLE_PROMPTS)
            print(f"\n[{elapsed:.1f}s] Query {query_count + 1}: '{query[:50]}...'")
            
            # Run inference
            t_start = time.time()
            result = process_input(query, components, dead_state)
            latency = (time.time() - t_start) * 1000.0
            
            # Let the shadow task run
            await asyncio.sleep(0.5)
            run_pending_timeline_a_shadows(components, dead_state)
            
            # Collect metrics from diagnostics
            latest_snap = None
            if components.inference_engine.diagnostics.history:
                latest_snap = components.inference_engine.diagnostics.history[-1]
                
            thermal = latest_snap.thermal_state if latest_snap else 0.0
            ram_headroom = latest_snap.ram_headroom_mb if latest_snap else 0.0
            x_next = latest_snap.x_used if latest_snap else configs.X_MAX
            
            print(f"  Timeline: {result['timeline']} | Latency: {latency:.1f} ms")
            print(f"  Thermal: {thermal:.1f}°C | RAM Headroom: {ram_headroom:.1f} MB | X_next: {x_next}")
            
            # Invariant checks
            if thermal > 85.0:
                print(f"  [warning] Thermal exceeds 85°C: {thermal:.1f}°C")
            if ram_headroom < 1024.0:
                print(f"  [warning] RAM headroom falls below 1GB: {ram_headroom:.1f} MB")
                
            record = {
                "elapsed_secs": elapsed,
                "timestamp": time.time(),
                "latency_ms": latency,
                "thermal": thermal,
                "ram_headroom_mb": ram_headroom,
                "x_next": x_next,
                "timeline": result['timeline'],
                "query": query,
            }
            log_records.append(record)
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
                
            query_count += 1
            # Wait for next query (taking elapsed time of inference into account)
            elapsed_query = time.time() - t_start
            sleep_time = max(0.1, query_interval - elapsed_query)
            await asyncio.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nStability test interrupted by user.")
    finally:
        orchestrator_task.cancel()
        try:
            await orchestrator_task
        except asyncio.CancelledError:
            pass
            
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  STABILITY TEST SUMMARY")
    print("=" * 80)
    print(f"  Total time run : {total_elapsed:.1f} seconds")
    print(f"  Total queries  : {query_count}")
    if log_records:
        thermals = [r["thermal"] for r in log_records if r["thermal"] > 0]
        ram_headrooms = [r["ram_headroom_mb"] for r in log_records if r["ram_headroom_mb"] > 0]
        x_nexts = [r["x_next"] for r in log_records]
        
        avg_thermal = sum(thermals)/len(thermals) if thermals else 0
        max_thermal = max(thermals) if thermals else 0
        min_ram = min(ram_headrooms) if ram_headrooms else 0
        
        print(f"  Avg Thermal    : {avg_thermal:.1f}°C")
        print(f"  Max Thermal    : {max_thermal:.1f}°C (Limit: 85°C)")
        print(f"  Min RAM Headroom: {min_ram:.1f} MB (Limit: 1024 MB)")
        print(f"  X_next Range   : {min(x_nexts)} to {max(x_nexts)}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sturnus stability and thermal check")
    parser.add_argument("--duration", type=int, default=7200, help="Test duration in seconds.")
    parser.add_argument("--interval", type=int, default=10, help="Seconds between queries.")
    args = parser.parse_args()
    
    try:
        asyncio.run(run_stability_test(args.duration, args.interval))
    except KeyboardInterrupt:
        pass
