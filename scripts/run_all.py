# pyre-unsafe
"""Run all steps of the Sturnus build."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
import data
import experts
import central
import gating
import main
from scripts import train_phase1, train_phase2, train_phase3
from scripts import validate


def run_all(live_data: bool, train: bool) -> None:
    print("[run_all] Step 2: config validation")
    configs.validate_config()

    print("[run_all] Step 3: data pipeline")
    data.self_test(live=live_data)

    print("[run_all] Step 4: experts")
    experts.self_test()

    print("[run_all] Step 5: central")
    central.self_test()

    print("[run_all] Step 6: gating")
    gating.self_test()

    print("[run_all] Step 7: main")
    main.self_test()

    if train:
        print("[run_all] Step 8: training")
        train_phase1.run()
        train_phase2.run()
        train_phase3.run()
    else:
        print("[run_all] Step 8: training skipped (use --train)")

    print("[run_all] Step 9: validation")
    validate.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    run_all(live_data=args.live_data, train=args.train)
