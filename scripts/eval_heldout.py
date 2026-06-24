"""CLI: evaluate the current checkpoint on the balanced held-out set.

  python scripts/eval_heldout.py                 # gate routing accuracy only (fast)
  python scripts/eval_heldout.py --full          # + held-out r_i / expert MSE (heavy)

Writes logs/heldout_eval.json. Use this as the honest, mixture-skew-immune
counterpart to the marathon's avg_loss.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from evaluation import (
    load_eval_set,
    gate_routing_accuracy,
    heldout_expert_quality,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Also run the heavy expert/Central held-out quality eval.")
    parser.add_argument("--eval-path", default="data/heldout_eval.jsonl")
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    samples = load_eval_set(args.eval_path)
    if not samples:
        print(f"[eval] no samples found at {args.eval_path}")
        return
    print(f"[eval] loaded {len(samples)} held-out prompts "
          f"({sum(1 for s in samples if s['domain']=='code')} code / "
          f"{sum(1 for s in samples if s['domain']=='reasoning')} reasoning / "
          f"{sum(1 for s in samples if s['domain']=='knowledge')} knowledge / "
          f"{sum(1 for s in samples if s['domain']=='general')} general)")

    from gating import GateModel, TripleKSelector, MaskingSchedule
    from apex_nadir_convolution import ApexNadirConvolution

    convolution = ApexNadirConvolution(configs.CALIBRATION_PATH, configs.LATENCY_STORE_PATH)
    convolution.load()
    gate = GateModel()
    gate.load()
    triple_k = TripleKSelector(convolution=convolution)
    masking = MaskingSchedule()

    t0 = time.time()
    gate_res = gate_routing_accuracy(gate, samples)
    gate_secs = time.time() - t0

    print("\n=== Gate routing accuracy (held-out, balanced) ===")
    print(f"  overall: {gate_res['accuracy']:.1%}  (n={gate_res['n']}, {gate_secs:.1f}s)")
    for d, acc in gate_res["per_domain"].items():
        print(f"    {d:10s}: {acc:.1%}")
    print("  confusion (row=true, col=predicted):")
    from evaluation import DOMAINS
    header = "            " + "".join(f"{d[:5]:>8s}" for d in DOMAINS)
    print(header)
    for td in DOMAINS:
        row = "".join(f"{gate_res['confusion'][td][pd]:>8d}" for pd in DOMAINS)
        print(f"    {td:10s}{row}")

    out = {"gate_routing": gate_res, "gate_eval_secs": gate_secs}

    if args.full:
        from experts import ExpertPool
        from central import CentralModel
        from memory import SessionTracker
        from splitter import get_available_ram_mb

        boot_ram = get_available_ram_mb()
        expert_budget = max(0, boot_ram - 4000)
        hw_max = min(max(1, int(expert_budget / configs.EXPERT_RAM_MB)), configs.K_MAX)
        session_tracker = SessionTracker()
        expert_pool = ExpertPool(convolution=convolution, session_tracker=session_tracker,
                                 max_loaded=max(2, hw_max))
        central = CentralModel()
        print(f"\n[eval] full mode: RAM={boot_ram:.0f}MB expert_cap={hw_max} — running expert/Central eval...")
        t1 = time.time()
        qual = heldout_expert_quality(
            gate, expert_pool, central, triple_k, masking, session_tracker,
            samples, k=args.k,
        )
        qual_secs = time.time() - t1
        print("\n=== Held-out expert quality ===")
        print(f"  mean r_i        : {qual['mean_r_i']:.4f}  (expert contribution; higher=better)")
        print(f"  mean expert MSE : {qual['mean_expert_mse']:.6f}  (to synthesis; lower=better)")
        print(f"  prompts used    : {qual['n']}  | expert activations: {qual['n_expert_activations']}  ({qual_secs:.1f}s)")
        out["expert_quality"] = qual
        out["expert_eval_secs"] = qual_secs

    Path("logs").mkdir(exist_ok=True)
    with open("logs/heldout_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[eval] wrote logs/heldout_eval.json")


if __name__ == "__main__":
    main()
