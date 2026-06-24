"""Realistic workload: repeated / paraphrased queries.

Synthetic non-repeating training data never triggers the Voronoi route cache, so
it can't show the central "gets faster with use" claim. This harness feeds a
realistic stream — a fixed set of unique queries, each in several surface
paraphrases, interleaved — and measures whether the routing memory actually
caches:

  Part A (routing layer, fast — gate only):
    * cluster hit rate over time (does it rise as the cache warms?)
    * routing latency on cache HIT vs MISS (hits skip expert re-selection)
    * cluster count growth

  Part B (--full, end-to-end — needs experts + Central):
    * routing overhead as a fraction of total inference latency
      (gate routing time / full pipeline time)

  python scripts/realistic_workload.py                 # Part A only (fast)
  python scripts/realistic_workload.py --full --n-full 8
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np

import configs

DOMAINS = ["code", "reasoning", "knowledge", "general"]

# 50 unique base queries (mix of domains). Repeated/paraphrased into a stream.
BASE_QUERIES = [
    ("code", "write a python function to reverse a string"),
    ("code", "how do I read a json file in python"),
    ("code", "implement binary search in java"),
    ("code", "what is a list comprehension in python"),
    ("code", "write a sql query to count rows per group"),
    ("code", "how do I handle exceptions in python"),
    ("code", "create a rest api endpoint with flask"),
    ("code", "explain async await in javascript"),
    ("code", "write a regex to match an email address"),
    ("code", "how do I merge two dictionaries in python"),
    ("code", "implement a stack using a linked list"),
    ("code", "what does the git rebase command do"),
    ("reasoning", "what is the derivative of x squared"),
    ("reasoning", "prove that there are infinitely many primes"),
    ("reasoning", "solve for x in 3x plus 5 equals 20"),
    ("reasoning", "what is the probability of rolling two sixes"),
    ("reasoning", "explain the chain rule in calculus"),
    ("reasoning", "find the area under the curve y equals x from 0 to 2"),
    ("reasoning", "what is the sum of the first 100 integers"),
    ("reasoning", "how do you compute a standard deviation"),
    ("reasoning", "explain bayes theorem with an example"),
    ("reasoning", "what is the limit of sin x over x as x goes to zero"),
    ("reasoning", "solve the quadratic equation x squared minus 4 equals 0"),
    ("reasoning", "explain the pigeonhole principle"),
    ("knowledge", "who painted the mona lisa"),
    ("knowledge", "what is the capital of france"),
    ("knowledge", "when did world war two end"),
    ("knowledge", "what is the speed of light"),
    ("knowledge", "who wrote romeo and juliet"),
    ("knowledge", "what is the largest planet in the solar system"),
    ("knowledge", "explain how photosynthesis works"),
    ("knowledge", "what causes earthquakes"),
    ("knowledge", "who discovered penicillin"),
    ("knowledge", "what is the boiling point of water"),
    ("knowledge", "describe the structure of an atom"),
    ("knowledge", "what year did the berlin wall fall"),
    ("general", "what should I have for breakfast"),
    ("general", "recommend a good movie to watch tonight"),
    ("general", "how do I stay focused while studying"),
    ("general", "suggest a gift for my mom's birthday"),
    ("general", "what are some fun weekend activities"),
    ("general", "help me plan a healthy weekly meal"),
    ("general", "how can I sleep better at night"),
    ("general", "give me tips for a job interview"),
    ("general", "what is a good morning routine"),
    ("general", "suggest a hobby I can start at home"),
    ("general", "how do I make small talk at a party"),
    ("general", "what's a good book to read on vacation"),
    ("general", "help me write a thank you note"),
    ("general", "how do I keep my houseplants alive"),
]

# Surface paraphrase templates — same intent, different gate fingerprint. This
# is the real test of the Voronoi similarity threshold (tau): do near-duplicate
# prompts map to the same cluster?
PARAPHRASE_TEMPLATES = [
    "{q}",
    "{q}?",
    "can you {q}",
    "please {q}",
    "{q}, thanks",
    "hey, {q}",
    "i was wondering {q}",
    "quick question: {q}",
    "could you help me — {q}",
    "{q} (be brief)",
]


def build_stream(repeats: int, seed: int = 7):
    rng = random.Random(seed)
    stream = []
    for base_id, (domain, q) in enumerate(BASE_QUERIES):
        for r in range(repeats):
            tmpl = PARAPHRASE_TEMPLATES[r % len(PARAPHRASE_TEMPLATES)]
            stream.append({"base_id": base_id, "domain": domain, "text": tmpl.format(q=q)})
    rng.shuffle(stream)
    return stream


def gate_domain_of(gate_out):
    logits = gate_out.domain_logits
    if logits is None or logits.shape[0] < len(DOMAINS):
        return "general"
    return DOMAINS[int(mx.argmax(logits[: len(DOMAINS)]).item())]


def part_a(repeats: int, k: int):
    from gating import GateModel, TripleKSelector, MaskingSchedule
    from memory import RoutingMemory, SessionTracker
    from apex_nadir_convolution import ApexNadirConvolution

    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    gate = GateModel(); gate.load()
    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()
    session_tracker = SessionTracker()
    routing_memory = RoutingMemory()   # fresh cache: watch it warm from cold

    stream = build_stream(repeats)
    print(f"[workload] {len(BASE_QUERIES)} unique queries x {repeats} paraphrases "
          f"= {len(stream)} prompts (shuffled)")

    hits = 0
    hit_lat, miss_lat = [], []
    cum_tokens = 0
    hit_flags = []          # 1 if cache hit, per prompt (in stream order)
    cluster_curve = []      # (#prompt, #clusters)
    base_seen_hit = {}      # base_id -> [hit?] to measure per-base repeat hit rate
    cluster_origin = {}     # cluster_id -> base_id that spawned it (correctness check)
    same_base_hits = 0      # hits whose matched cluster came from the SAME base query
    same_domain_hits = 0    # hits whose matched cluster shares the query's domain

    for i, item in enumerate(stream):
        token_ids = gate.tokenizer.encode(item["text"])[: configs.MAX_SEQ_LEN]
        if len(token_ids) < 1:
            continue
        tokens = mx.array(token_ids)
        cum_tokens += len(token_ids)

        t0 = time.time()
        gate_out, _topo = gate.forward_with_topography(tokens)
        hit = routing_memory.lookup(gate_out.hidden_states)
        if hit is not None:
            route_ms = (time.time() - t0) * 1000.0   # HIT: cache supplies experts, skip selection
            hit_lat.append(route_ms)
            hits += 1
            hit_flags.append(1)
            base_seen_hit.setdefault(item["base_id"], []).append(1)
            # Correctness: did we hit a cluster spawned by the SAME query (or at
            # least the same domain), or is tau so loose it matches anything?
            if cluster_origin.get(hit.cluster_id) == item["base_id"]:
                same_base_hits += 1
            if getattr(hit, "domain", None) == item["domain"]:
                same_domain_hits += 1
        else:
            selected = triple_k.select_experts(gate_out, session_tracker, masking, i)
            route_ms = (time.time() - t0) * 1000.0   # MISS: paid full expert selection
            miss_lat.append(route_ms)
            hit_flags.append(0)
            base_seen_hit.setdefault(item["base_id"], []).append(0)
            domain = gate_domain_of(gate_out)
            expert_ids = [s.expert_id for s in selected][:k]
            # Spawn so future similar prompts can hit (force r_i above the floor —
            # we are measuring the lookup/similarity mechanism, not r_i gating).
            cl = routing_memory.spawn_cluster(
                gate_hidden=gate_out.hidden_states,
                expert_ids=expert_ids,
                tkl_scores={e: 1.0 for e in expert_ids},
                r_out_snapshot={}, l_eff_scores={},
                optimal_k=max(1, len(expert_ids)),
                token_count=cum_tokens, r_i=1.0, domain_mean_r_i=0.0, domain=domain,
            )
            if cl is not None:
                cluster_origin[cl.cluster_id] = item["base_id"]
        cluster_curve.append((i + 1, len(routing_memory.clusters)))

    n = len(hit_flags)
    half = n // 2
    first_half_hit = float(np.mean(hit_flags[:half])) if half else 0.0
    second_half_hit = float(np.mean(hit_flags[half:])) if (n - half) else 0.0
    # per-base: of repeats AFTER the first appearance, how many hit?
    repeat_hits, repeat_total = 0, 0
    for flags in base_seen_hit.values():
        for f in flags[1:]:
            repeat_total += 1
            repeat_hits += f

    same_base_acc = same_base_hits / max(hits, 1)
    same_domain_acc = same_domain_hits / max(hits, 1)
    print("\n=== Part A — routing memory (Voronoi cache) ===")
    print(f"  prompts processed     : {n}")
    print(f"  clusters formed       : {len(routing_memory.clusters)}")
    print(f"  overall hit rate      : {hits}/{n} = {hits/max(n,1):.1%}")
    print(f"  hit rate 1st half     : {first_half_hit:.1%}")
    print(f"  hit rate 2nd half     : {second_half_hit:.1%}   (warming = should rise)")
    print(f"  repeat-only hit rate  : {repeat_hits}/{max(repeat_total,1)} = {repeat_hits/max(repeat_total,1):.1%}  "
          f"(of non-first appearances)")
    print(f"  HIT correctness:")
    print(f"    same-base hits      : {same_base_hits}/{hits} = {same_base_acc:.1%}  "
          f"(matched the SAME query's cluster — real recognition)")
    print(f"    same-domain hits    : {same_domain_hits}/{hits} = {same_domain_acc:.1%}  "
          f"(matched a same-domain cluster)")
    if same_base_acc < 0.5:
        print(f"    ⚠ low same-base accuracy → tau likely too loose (matches unrelated prompts)")
    if hit_lat:
        print(f"  routing latency HIT   : {np.mean(hit_lat):.1f} ms  (gate + lookup, no selection)")
    if miss_lat:
        print(f"  routing latency MISS  : {np.mean(miss_lat):.1f} ms  (gate + lookup + selection + spawn)")
    if hit_lat and miss_lat:
        print(f"  cache speedup on hits : {np.mean(miss_lat)/max(np.mean(hit_lat),1e-6):.2f}x")

    return {
        "prompts": n,
        "clusters": len(routing_memory.clusters),
        "overall_hit_rate": hits / max(n, 1),
        "hit_rate_first_half": first_half_hit,
        "hit_rate_second_half": second_half_hit,
        "repeat_hit_rate": repeat_hits / max(repeat_total, 1),
        "same_base_accuracy": same_base_acc,
        "same_domain_accuracy": same_domain_acc,
        "routing_ms_hit": float(np.mean(hit_lat)) if hit_lat else None,
        "routing_ms_miss": float(np.mean(miss_lat)) if miss_lat else None,
        "cluster_curve": cluster_curve[::max(1, n // 40)],
    }


def part_b(n_full: int, k: int):
    """End-to-end routing overhead %, on a small sample. Times the routing
    segment (fused gate pass + expert selection) vs the full pipeline
    (+ expert forwards + Central synthesis). Mirrors the Timeline-B path."""
    from gating import GateModel, TripleKSelector, MaskingSchedule
    from memory import SessionTracker
    from apex_nadir_convolution import ApexNadirConvolution
    from experts import ExpertPool
    from central import CentralModel
    from splitter import get_available_ram_mb

    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    gate = GateModel(); gate.load()
    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()
    session_tracker = SessionTracker()
    boot_ram = get_available_ram_mb()
    hw_max = min(max(1, int(max(0, boot_ram - 4000) / configs.EXPERT_RAM_MB)), configs.K_MAX)
    expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker, max_loaded=max(2, hw_max))
    central = CentralModel(); central.load()

    sample = [q for q in BASE_QUERIES][:n_full]
    print(f"\n[workload] Part B — end-to-end routing overhead on {len(sample)} queries "
          f"(expert_cap={hw_max})")

    routing_times, total_times = [], []
    for domain, q in sample:
        token_ids = gate.tokenizer.encode(q)[: configs.MAX_SEQ_LEN]
        if len(token_ids) < configs.FRAGMENT_MIN:
            token_ids = (token_ids * (configs.FRAGMENT_MIN // max(len(token_ids), 1) + 1))[: configs.FRAGMENT_MIN]
        tokens = mx.array(token_ids)
        n_tokens = len(token_ids)

        t_start = time.time()
        # --- routing segment ---
        t_r = time.time()
        gate_out, _topo = gate.forward_with_topography(tokens)
        selected = triple_k.select_experts(gate_out, session_tracker, masking, 0)[:k]
        routing_ms = (time.time() - t_r) * 1000.0

        # --- experts + central segment ---
        ids_to_load = [s.expert_id for s in selected if s.expert_id not in expert_pool.loaded_experts]
        try:
            if ids_to_load:
                expert_pool.load_experts(ids_to_load)
        except RuntimeError:
            continue
        selected = [s for s in selected if s.expert_id in expert_pool.loaded_experts]
        if not selected:
            continue
        frag = max(configs.FRAGMENT_MIN, n_tokens // max(len(selected), 1))
        eos = []
        for i, sel in enumerate(selected):
            fs, fe = i * frag, min(i * frag + frag, n_tokens)
            if fs >= n_tokens:
                break
            ft = tokens[fs:fe]
            if ft.shape[0] < configs.FRAGMENT_MIN:
                continue
            eos.append(expert_pool.expert_forward(sel.expert_id, ft))
        if not eos:
            continue
        expert_data = [{"expert_id": e.expert_id, "output_text": e.output_text,
                        "hidden_states": e.hidden_states, "wall_time": e.wall_time} for e in eos]
        cout = central.forward(q, expert_data, send_to_user=False)
        mx.eval(cout.synthesis_hidden, cout.contribution_hidden)
        total_ms = (time.time() - t_start) * 1000.0
        routing_times.append(routing_ms)
        total_times.append(total_ms)

    if not total_times:
        print("  (no successful end-to-end samples)")
        return {}
    mean_route = float(np.mean(routing_times))
    mean_total = float(np.mean(total_times))
    overhead = mean_route / mean_total
    print("\n=== Part B — end-to-end routing overhead ===")
    print(f"  mean routing time : {mean_route:.1f} ms  (fused gate pass + expert selection)")
    print(f"  mean total time   : {mean_total:.1f} ms  (+ experts + Central synthesis)")
    print(f"  routing overhead  : {overhead:.1%}   (target < 15%)")
    return {"routing_ms": mean_route, "total_ms": mean_total, "routing_overhead": overhead,
            "n": len(total_times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10, help="paraphrases per unique query")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="also measure end-to-end routing overhead")
    ap.add_argument("--n-full", type=int, default=8, help="queries for the end-to-end measurement")
    args = ap.parse_args()

    out = {"part_a": part_a(args.repeats, args.k)}
    if args.full:
        out["part_b"] = part_b(args.n_full, args.k)

    Path("logs").mkdir(exist_ok=True)
    with open("logs/realistic_workload.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[workload] wrote logs/realistic_workload.json")


if __name__ == "__main__":
    main()
