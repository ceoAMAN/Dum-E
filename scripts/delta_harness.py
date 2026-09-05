"""Δ HARNESS — the acceptance gate for the whole redesign.

Δ is the grounded contribution of one expert:

    Δ = CE(Central(question),                y_true)
      − CE(Central(question + expert_text),  y_true)

Positive means the expert made the REAL continuation more predictable. The
referent is text neither model wrote, which is the only thing in the system
that can contradict the models.

This harness measures Δ against three deliberately different expert outputs and
compares it, side by side, with the cosine r_i the engine uses today.

ACCEPTANCE — all three must hold, or nothing downstream is worth building:

    1. Δ(true) > Δ(irrelevant)          the signal ranks correctly
    2. Δ(true) > Δ(confidently wrong)   nonsense is punished, not rewarded
    3. spread(Δ) >> spread(cosine)      Δ discriminates where cosine does not

Run:  python scripts/delta_harness.py
      python scripts/delta_harness.py --no-cosine   (skip the expert load)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import configs
from central import CentralModel
from training import central_ce_value

# ── cases ────────────────────────────────────────────────────────────────────
# Each: a question, its REAL continuation (the ground truth Δ is scored against),
# and three expert analyses — one true and relevant, one off-topic, one stated
# with confidence and wrong. The third is the one that matters: cosine rates it
# as highly as the true one, because confident text moves Central a lot and
# cosine rewards movement rather than correctness.
CASES = [
    {
        "domain": "code",
        "question": "Why is quicksort O(n log n) on average but O(n^2) in the worst case?",
        "truth": (
            " On average the pivot splits the array into two roughly equal parts, so the "
            "recursion depth is log n and each level costs n comparisons. In the worst case "
            "the pivot is always the smallest or largest element, so one partition is empty "
            "and the recursion depth becomes n, giving n levels of n comparisons."
        ),
        "experts": {
            "true": (
                "Key insight: the cost is set by how evenly the pivot partitions the array. "
                "Balanced splits give recursion depth log n; a pivot that is always the "
                "extreme element gives depth n, and each level still costs n comparisons."
            ),
            "irrelevant": (
                "Key insight: the Arctic tern migrates roughly 70,000 km each year between "
                "its Arctic breeding grounds and the Antarctic, the longest known migration "
                "of any animal."
            ),
            "wrong": (
                "Key insight: quicksort is O(n log n) because it uses a divide-and-conquer "
                "merge step that always splits the input exactly in half, and the worst case "
                "is O(n^2) because merging two sorted halves requires comparing every element "
                "against every other element."
            ),
        },
    },
    {
        "domain": "reasoning",
        "question": "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
        "truth": (
            " 45 minutes is 0.75 hours. Average speed is distance divided by time, so "
            "60 divided by 0.75 gives 80 km/h."
        ),
        "experts": {
            "true": (
                "Key insight: convert the time to hours first — 45 minutes is 0.75 hours — "
                "then divide distance by time."
            ),
            "irrelevant": (
                "Key insight: the Baroque period in European music is usually dated from "
                "1600 to 1750 and ends with the death of Johann Sebastian Bach."
            ),
            "wrong": (
                "Key insight: average speed is distance multiplied by time, so multiply 60 km "
                "by 45 minutes to get the answer directly in km/h."
            ),
        },
    },
    {
        "domain": "knowledge",
        "question": "What causes the seasons on Earth?",
        "truth": (
            " The seasons are caused by the tilt of Earth's rotational axis, about 23.5 "
            "degrees relative to its orbital plane. This tilt means each hemisphere receives "
            "sunlight at a more direct angle for part of the year and a shallower angle for "
            "the rest, changing the energy received per unit area."
        ),
        "experts": {
            "true": (
                "Key insight: the driver is the 23.5 degree axial tilt, which changes the "
                "angle at which sunlight strikes each hemisphere over the course of the orbit."
            ),
            "irrelevant": (
                "Key insight: a standard chessboard has 64 squares arranged in an eight by "
                "eight grid, alternating between light and dark."
            ),
            "wrong": (
                "Key insight: the seasons are caused by Earth's elliptical orbit changing its "
                "distance from the Sun — we are closest in summer and furthest in winter, "
                "which is why summer is warmer."
            ),
        },
    },
    {
        "domain": "general",
        "question": "Why does bread rise when you bake it?",
        "truth": (
            " Yeast ferments the sugars in the dough and releases carbon dioxide, which is "
            "trapped by the gluten network. In the oven the trapped gas expands as it heats, "
            "the water turns to steam, and the dough sets as the proteins coagulate and the "
            "starches gelatinise, locking the expanded structure in place."
        ),
        "experts": {
            "true": (
                "Key insight: carbon dioxide from yeast fermentation is trapped by gluten; "
                "heat expands that trapped gas and turns water to steam before the proteins "
                "set the structure."
            ),
            "irrelevant": (
                "Key insight: the Danish krone has been pegged to the euro within a narrow "
                "band since the introduction of the single currency."
            ),
            "wrong": (
                "Key insight: bread rises because baking powder reacts with the salt in the "
                "dough to produce oxygen, and oxygen is lighter than air so the loaf floats "
                "upward as it bakes."
            ),
        },
    },
]

LABELS = ("true", "irrelevant", "wrong")


def measure_delta(central, question: str, target_ids, expert_text: str):
    """Δ for a single expert output: CE without it minus CE with it."""
    without = central_ce_value(central, question, [], target_ids)
    with_it = central_ce_value(central, question, [{"output_text": expert_text}], target_ids)
    if without is None or with_it is None:
        return None, None, None
    return float(without - with_it), float(without), float(with_it)


def measure_cosine(central, expert_pool, expert_id: int, question: str, expert_text: str):
    """The cosine r_i the engine uses today, for the same expert output.

    Runs the expert over its own analysis to get a hidden state, then scores it
    against Central's contribution/synthesis vectors — exactly what
    _timeline_b does.
    """
    import mlx.core as mx

    out = central.forward(question, [{"output_text": expert_text}], send_to_user=False)
    tok = expert_pool.loaded_tokenizers[expert_id]
    eo = expert_pool.expert_forward(
        expert_id, mx.array(tok.encode(expert_text)),
        generate_text=False, allocated_tokens=32,
    )
    return float(central.compute_r_i(
        eo.hidden_states, out.contribution_hidden, 0.5,
        synthesis_hidden=out.synthesis_hidden,
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cosine", action="store_true",
                    help="skip the cosine comparison (avoids loading an expert)")
    ap.add_argument("--expert", type=int, default=9, help="expert id for the cosine baseline")
    args = ap.parse_args()

    central = CentralModel()
    central.load()
    tok = central.tokenizer

    expert_pool = None
    if not args.no_cosine:
        from apex_nadir_convolution import ApexNadirConvolution
        from experts import ExpertPool
        from memory import SessionTracker
        conv = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
        conv.load()
        expert_pool = ExpertPool(convolution=conv, session_tracker=SessionTracker())
        expert_pool.load_experts([args.expert])

    deltas = {k: [] for k in LABELS}
    cosines = {k: [] for k in LABELS}

    for case in CASES:
        q, truth = case["question"], case["truth"]
        target_ids = tok.encode(truth)
        print(f"\n{'─' * 74}\n{case['domain']}: {q}")
        print(f"{'label':<12}{'Δ':>10}{'CE_without':>13}{'CE_with':>10}"
              + ("" if args.no_cosine else f"{'cosine r_i':>13}"))
        for label in LABELS:
            text = case["experts"][label]
            d, without, with_it = measure_delta(central, q, target_ids, text)
            if d is None:
                print(f"{label:<12}{'—':>10}   (no target region)")
                continue
            deltas[label].append(d)
            row = f"{label:<12}{d:>+10.4f}{without:>13.4f}{with_it:>10.4f}"
            if not args.no_cosine:
                c = measure_cosine(central, expert_pool, args.expert, q, text)
                cosines[label].append(c)
                row += f"{c:>13.4f}"
            print(row)

    # ── verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 74}\nMEANS OVER {len(CASES)} CASES\n{'═' * 74}")
    md = {k: float(np.mean(v)) if v else float("nan") for k, v in deltas.items()}
    mc = {k: float(np.mean(v)) if v else float("nan") for k, v in cosines.items()}

    print(f"{'label':<14}{'Δ (grounded)':>16}" + ("" if args.no_cosine else f"{'cosine r_i':>16}"))
    for label in LABELS:
        row = f"{label:<14}{md[label]:>+16.4f}"
        if not args.no_cosine:
            row += f"{mc[label]:>16.4f}"
        print(row)

    d_spread = max(md.values()) - min(md.values())
    print(f"\n{'Δ spread':<14}{d_spread:>16.4f}")
    if not args.no_cosine:
        c_spread = max(mc.values()) - min(mc.values())
        print(f"{'cosine spread':<14}{c_spread:>16.4f}")
        ratio = d_spread / c_spread if c_spread > 1e-9 else float("inf")
        print(f"{'ratio':<14}{ratio:>16.1f}x")

    checks = [
        ("1. Δ(true) > Δ(irrelevant)", md["true"] > md["irrelevant"]),
        ("2. Δ(true) > Δ(wrong)", md["true"] > md["wrong"]),
    ]
    if not args.no_cosine:
        c_spread = max(mc.values()) - min(mc.values())
        checks.append(("3. Δ spread > 5x cosine spread",
                       d_spread > 5 * c_spread if c_spread > 1e-9 else True))

    print(f"\n{'─' * 74}")
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok = ok and passed

    print(f"\n{'GATE OPEN — Δ discriminates; build the rest.' if ok else 'GATE CLOSED — Δ does not separate. Do not build on it.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
