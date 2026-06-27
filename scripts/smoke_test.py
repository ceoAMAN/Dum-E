"""Engine smoke test: boot the REAL models (gate 0.5B + experts 1.5B + central
7B) and run a few real training batches through run_marathon, with the HuggingFace
data pipeline replaced by synthetic in-memory samples so it needs no token, no
network, and no dataset download.

This exercises exactly the paths that pure-unit tests can't: gate.load() building
GateNet, value_and_grad over the routing head on the real backbone, real expert
text generation, central.forward synthesis, dual R_i / L_eff / peer repulsion, and
checkpoint save (LoRA + route_head + apex curves).

Run:  /Users/aman/brian-env/bin/python scripts/smoke_test.py --batches 10
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
import data
from data import Sample

# Diverse prompts long enough to clear FRAGMENT_MIN and split across >=2 experts
# (so peer repulsion actually runs). Four domains so routing/L_dom see variety.
_BASE = {
    "code": "Write a Python function that merges two sorted lists into one sorted list. "
            "def merge(a, b): create an empty result, use two index pointers i and j, "
            "while both lists have elements compare a[i] and b[j], append the smaller and "
            "advance its pointer, then append the remainder. Explain the time complexity "
            "and why it is O(n + m), and give an example call with [1,3,5] and [2,4,6]. ",
    "reasoning": "Prove that the sum of the first n odd numbers equals n squared. "
                 "Consider 1 + 3 + 5 + ... + (2n-1). Use induction: base case n=1 gives 1 = 1^2. "
                 "Assume true for k, then the sum for k+1 adds (2k+1), giving k^2 + 2k + 1 = (k+1)^2. "
                 "Therefore by induction the statement holds for all positive integers n. ",
    "knowledge": "Explain how photosynthesis converts light energy into chemical energy. "
                 "Describe the light-dependent reactions in the thylakoid membrane, the role "
                 "of chlorophyll, the electron transport chain, ATP and NADPH production, and "
                 "the Calvin cycle in the stroma where carbon dioxide is fixed into glucose. ",
    "general": "Describe a calm morning routine that helps someone feel focused and ready for "
               "the day. Talk about waking early, light stretching, a glass of water, a few "
               "minutes of quiet planning, and a simple breakfast, and why small consistent "
               "habits compound into steady energy and a clearer mind over time. ",
}
# Repeat each so token count is comfortably above the bootstrap fragment size.
SMOKE_PROMPTS = [v * 3 for v in _BASE.values()]


def fake_iter_mixture_samples(seed: int = 42):
    i = 0
    while True:
        text = SMOKE_PROMPTS[i % len(SMOKE_PROMPTS)]
        yield Sample(source="smoke", text=text, raw={"text": text})
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=10)
    args = ap.parse_args()

    # Bypass the HF data pipeline entirely.
    configs.HF_TOKEN = "smoke"                       # satisfy validate_config
    data.iter_mixture_samples = fake_iter_mixture_samples
    import scripts.finetune as ft
    ft.iter_mixture_samples = fake_iter_mixture_samples   # finetune bound the name at import
    ft.authenticate_huggingface = lambda: print("[smoke] HF auth skipped (synthetic data)")

    print(f"[smoke] booting real models + running {args.batches} training batches...")
    ft.run_marathon(
        max_tokens=10 ** 12,
        checkpoint_every=max(1, args.batches // 2),
        print_every=1,
        clean=True,
        max_batches=args.batches,
    )
    print("[smoke] PASS — engine booted and trained for the requested batches without error.")


if __name__ == "__main__":
    main()
