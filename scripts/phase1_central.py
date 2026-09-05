"""PHASE 1 — Central trains alone on T/2 tokens, then is frozen for the joint
phase. Trained FIRST so that r_i, which is measured against Central's loss,
stays comparable across the whole run: a Central that keeps moving is a
yardstick that keeps moving."""
import sys, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import configs, data, main
from central import CentralModel

ap = argparse.ArgumentParser()
ap.add_argument("--total-tokens", type=int, default=200_000, help="T; Central gets T/2")
ap.add_argument("--print-every", type=int, default=25)
args = ap.parse_args()

data.authenticate_huggingface()
central = CentralModel(); central.load()
tok = central.tokenizer

def samples():
    """(question, real continuation) pairs. The continuation is the GROUND TRUTH
    the CE is scored against — the only non-self-referential signal in the system."""
    for s in data.iter_mixture_samples(seed=42):
        ids = tok.encode(s.text)[: configs.MAX_SEQ_LEN]
        if len(ids) < 32:
            continue
        cut = max(16, int(len(ids) * 0.7))
        yield tok.decode(ids[:cut]), ids[cut:]

class C: pass
c = C(); c.central = central
t0 = time.time()
stats = main.pretrain_central(c, samples(), total_token_budget=args.total_tokens,
                              print_every=args.print_every)
print(f"\n[phase1] {stats['steps']} steps · {stats['tokens']} tokens · "
      f"ce {stats['ce_first']:.4f} -> {stats['ce_last']:.4f} · {time.time()-t0:.0f}s")
print(f"[phase1] Central saved to {configs.CHECKPOINT_DIR}/central/")
