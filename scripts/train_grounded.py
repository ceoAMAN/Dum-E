"""Grounded training loop — the caller that makes target_ids exist.

Every quality signal in this system is supposed to be anchored to text neither
the gate, the experts, nor Central wrote. The machinery for that was all
present and all dead:

    data.extract_pair          produced prompt/answer for 8/8 datasets — no consumer
    InferenceEngine.run        accepted target_ids — no caller ever passed them
    _grounded_r_i_override     guarded by `if target_ids:` — permanently False
    training.grounded_r_i      callerless on the engine path
    central.compute_grounded_r_i   never reached
    the Central capacity re-probe  needs 2 samples, could never get 1
    compute_r_i_batch's "perfection" component   loss_deltas had no supplier

Nothing was broken. Every piece worked. There was simply no path from a dataset
answer to the code that scores against one. This script is that path.

    python scripts/train_grounded.py --steps 20
    python scripts/train_grounded.py --steps 200 --verifiable-only

What it does per step: pull a sample that HAS a ground-truth answer, tokenise
that answer with Central's tokenizer, and run the normal Timeline B pipeline
with it. Everything downstream then behaves as designed — r_i becomes the
leave-one-out delta against real text instead of cosine agreement with Central's
own hidden state, and the gate, the experts and the clusters all learn from that
instead of from a circular reward.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import configs
import main as dume
from data import iter_mixture_samples


def run(steps: int, verifiable_only: bool, max_answer_tokens: int, seed: int,
        report_every: int) -> int:
    components = dume.boot_system()
    engine = components.inference_engine
    central = components.central
    central.load()                      # need the tokenizer, and it loads lazily

    stream = iter_mixture_samples(seed=seed)
    done = skipped = 0
    deltas_seen = 0
    t0 = time.time()

    print(f"[train] grounded loop: {steps} steps, "
          f"{'verifiable answers only' if verifiable_only else 'any ground-truth answer'}, "
          f"scoring {configs.GROUNDED_SAMPLE_K} expert(s) per batch against real text")

    while done < steps:
        try:
            sample = next(stream)
        except StopIteration:
            print("[train] mixture stream exhausted")
            break
        except Exception as e:
            print(f"[train] stream error, continuing: {e}")
            continue

        # A sample without an answer is useless here — the whole point is the
        # external referent. Skip rather than fabricate one.
        if not sample.has_target:
            skipped += 1
            continue
        if verifiable_only and not sample.has_verifiable:
            skipped += 1
            continue

        prompt = sample.prompt or sample.text
        target_ids = central.tokenizer.encode(sample.answer)[:max_answer_tokens]
        if not target_ids:
            skipped += 1
            continue

        try:
            # force_timeline_b: the fast path skips experts entirely, and an
            # expert that never runs cannot be scored against the ground truth.
            result = engine.run(prompt, send_to_user=False, force_timeline_b=True,
                                target_ids=target_ids)
        except Exception as e:
            print(f"[train] step {done + 1} failed ({type(e).__name__}: {e}); continuing")
            skipped += 1
            continue

        done += 1
        if getattr(engine, "_grounded_batches", 0) > deltas_seen:
            deltas_seen = engine._grounded_batches

        if done % report_every == 0:
            rate = done / max(1e-9, time.time() - t0)
            print(f"[train] {done}/{steps} steps | {deltas_seen} grounded scorings | "
                  f"{skipped} skipped | mean r_i {getattr(result, 'mean_r_i', 0.0):.3f} | "
                  f"{rate * 60:.1f} steps/min")

    dume.session_reset(components, dume.DeadTimeState())
    print(f"[train] done: {done} steps, {deltas_seen} batches scored against real text, "
          f"{skipped} samples skipped, {time.time() - t0:.0f}s")
    if done and not deltas_seen:
        # Loud, because this is the exact shape of the bug this script exists to
        # fix: the loop ran, nothing errored, and no grounded measurement happened.
        print("[train] WARNING: not one grounded scoring occurred despite running "
              "steps — target_ids are not reaching _grounded_r_i_override")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--verifiable-only", action="store_true",
                    help="Only use samples whose answer is externally checkable "
                         "(gsm8k, MetaMathQA, ai2_arc, sciq).")
    ap.add_argument("--max-answer-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-every", type=int, default=1)
    a = ap.parse_args()
    sys.exit(run(a.steps, a.verifiable_only, a.max_answer_tokens, a.seed, a.report_every))
