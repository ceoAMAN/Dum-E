from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import configs
from data import authenticate_huggingface
from scripts import train_phase1, train_phase2, train_phase3
from scripts import validate

def run_all(train: bool) -> None:
    print("[run_all] Step 1: config validation")
    configs.validate_config()
    print("[run_all] Config OK")
    print("[run_all] Step 2: HuggingFace auth")
    authenticate_huggingface()
    print("[run_all] Auth OK")
    if train:
        print("[run_all] Step 3: Phase 1 — Central fine-tuning")
        train_phase1.run()
        print("[run_all] Step 4: Phase 2 — Gate fine-tuning")
        train_phase2.run()
        print("[run_all] Step 5: Phase 3 — Expert fine-tuning")
        train_phase3.run()
    else:
        print("[run_all] Steps 3-5: training skipped (use --train)")
    print("[run_all] Step 6: validation")
    validate.run()
    print("[run_all] All steps complete")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    run_all(train=args.train)
