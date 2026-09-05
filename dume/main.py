"""CLI.

  python -m dume.main form     --samples 200
  python -m dume.main pretrain --tokens 20000
  python -m dume.main train    --batches 50
  python -m dume.main run      --prompt "..."
  python -m dume.main status
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as C
from .train import System


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dume")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("form");     f.add_argument("--samples", type=int, default=200)
    p = sub.add_parser("pretrain"); p.add_argument("--tokens", type=int, default=20000)
    t = sub.add_parser("train");    t.add_argument("--batches", type=int, default=50)
    r = sub.add_parser("run");      r.add_argument("--prompt", required=True); r.add_argument("--max-tokens", type=int, default=256)
    sub.add_parser("status")
    a = ap.parse_args(argv)

    if a.cmd == "status":
        from . import state
        blob = state.load()
        if not blob:
            print("cold"); return 0
        g = blob["geometry"]; st = blob["standing"]
        print(json.dumps({"clusters": int(g["B"].shape[0]), "tau": [round(float(x), 3) for x in g["tau"]],
                          "version": g["version"], "clock": blob["clock"], "batch": blob["batch"],
                          "standing_n": float(st["n"].sum()), "moves": st["moves"],
                          "reliability_obs": float(blob["reliability"].total_obs()),
                          "reliability": [round(float(x), 3) for x in blob["reliability"].vector()]}, indent=1))
        return 0

    sysm = System().boot(need_experts=(a.cmd in ("train", "run")))
    if a.cmd == "form":
        g = sysm.form(a.samples)
        print(json.dumps({"clusters": g.C, "tau": [round(float(x), 3) for x in g.tau], "version": g.version}))
    elif a.cmd == "pretrain":
        print(json.dumps(sysm.pretrain(a.tokens)))
    elif a.cmd == "train":
        out = sysm.train(a.batches)
        print(json.dumps(out))
        if out["measured_this_run"] <= 0:
            print("[train] FAILED: this run recorded no grounded measurement — y never arrived", file=sys.stderr)
            return 1
    elif a.cmd == "run":
        out = sysm.answer(a.prompt, a.max_tokens)
        print(json.dumps({k: v for k, v in out.items() if k != "text"}, indent=1))
        print("\n" + out["text"])
        sysm.save()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code or 0))     # HF streaming can leave a non-daemon thread; do not let it hold exit
