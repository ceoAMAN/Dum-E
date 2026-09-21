#!/usr/bin/env python3
"""Did every mechanism actually fire?

A mechanism can be correct in every line, log clean records for thousands of
batches, pass every assertion in dume_check.py, and do nothing. The thermal
regulator did exactly that: `NSProcessInfo.thermalState()` returned `fair` on
all 580 samples of a 2900-batch run, so its baseline converged to the one value
it ever saw, its pressure was identically zero, and k_thermal sat at k_max for
the entire run. Nothing was wrong. The device simply never voted.

Nothing greps for that. The only thing that catches it is asking a log whether a
number ever CHANGED, which is what this does.

Two things make it sharp.

THE TAIL. Movement alone is not the test. In the very run that hid this bug
k_thermal DID move -- 2.758 -> 4.000 during warmup, as its baseline converged
onto the one level it would ever see -- and then never moved again for 2900
batches. A mechanism that settles once and flatlines is inert no matter how
lively it looked at the start, so every field is judged on the LAST HALF of the
run as well as the whole of it.

THE CHAIN. A constant output is not by itself a bug -- a cool machine should
produce no thermal pressure. It is a bug when the output is constant while its
INPUT moved, because then the mechanism had something to say and said nothing.
When the input is dead too, the fault is upstream at the sensor, which is where
it actually was.

    python3 scripts/did_it_fire.py                     # the live run
    python3 scripts/did_it_fire.py logs/archive/*/     # an archived one
"""
from __future__ import annotations

import glob
import re
import sys
from collections import OrderedDict

# a mechanism and the input it is supposed to answer to. If the output never
# moves, the input tells you WHICH bug you have.
CHAINS = [
    ("thermal", "thermal_z"),          # die degrees -> excursion above normal
    ("thermal_z", "thermal_p"),        # excursion -> pressure (deadband may eat it)
    ("thermal_p", "k_thermal"),        # pressure -> the device's vote on k
    ("run_left", "k_thermal"),         # how much run is left scales that vote
    ("k_wanted", "k"),                 # the allocation law -> the k actually run
    ("rho", "trust"),                  # routing confidence -> what it is trusted for
]
# structural: these move or the run is not a run
MUST_MOVE = ["clock", "standing_n", "reliability_obs", "run_left",
             "gate_loss", "central_loss"]

LINE = re.compile(r"\[health b(\d+)\]\s*(.*)")


def read(paths):
    """{key: [values]} from every `[health bN]` line in these logs, in order."""
    seen, cols = set(), OrderedDict()
    for p in sorted(paths):
        for line in open(p, errors="replace"):
            m = LINE.match(line.strip())
            if not m or (p, m.group(1)) in seen:
                continue
            seen.add((p, m.group(1)))
            for kv in m.group(2).split():
                k, _, v = kv.partition("=")
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    pass
    return cols


def alive(vals):
    """Distinct values over the LAST HALF. A field that settled during warmup and
    then flatlined is dead, however much it moved getting there."""
    return len(set(vals[len(vals) // 2:])) > 1 if vals else False


def main(argv):
    args = argv[1:] or ["logs"]
    paths = [f for a in args for f in (glob.glob(a + "/*.log") if "." not in a.split("/")[-1]
                                       else glob.glob(a))]
    cols = read(paths)
    if not cols:
        print(f"no [health bN] records in {args}")
        return 2
    n = max(len(v) for v in cols.values())
    print(f"{n} health records from {len(paths)} log(s)\n")
    print(f"{'field':24s} {'n':>5} {'distinct':>9} {'min':>12} {'max':>12}   verdict")
    dead = []
    for k, v in sorted(cols.items()):
        d = len(set(v))
        verdict = "moved" if alive(v) else ("WARMUP ONLY" if d > 1 else "CONSTANT")
        if not alive(v):
            dead.append(k)
        print(f"{k:24s} {len(v):5d} {d:9d} {min(v):12.4f} {max(v):12.4f}   {verdict}")

    # a run that has barely started cannot distinguish "inert" from "has not had
    # the chance". run_left says how far in it is, so the note is measured.
    done = 1.0 - cols["run_left"][-1] if cols.get("run_left") else None
    if done is not None and done < 0.5:
        print(f"NOTE: the run is {100 * done:.1f}% done. A field flagged below may simply "
              f"not have had the chance yet — re-run this at the end.\n")
    bad = 0
    for src, dst in CHAINS:
        if dst not in cols:
            print(f"  {src:14s} -> {dst:14s}  ABSENT: never recorded")
            bad += 1
            continue
        moved_in = src in cols and alive(cols[src])
        moved_out = alive(cols[dst])
        if moved_out:
            print(f"  {src:14s} -> {dst:14s}  ok")
        elif moved_in:
            print(f"  {src:14s} -> {dst:14s}  INERT: the input moved and the output did not")
            bad += 1
        else:
            print(f"  {src:14s} -> {dst:14s}  input dead too -- look UPSTREAM of {src}")
            bad += 1

    missing = [k for k in MUST_MOVE if k not in cols or not alive(cols[k])]
    if missing:
        print(f"\n  structural fields that never moved: {', '.join(missing)}")
        bad += len(missing)
    print(f"\n{len(dead)} field(s) dead in the run's second half; {bad} problem(s)")
    return 1 if bad else 0


def _self_check():
    """The 2900-batch thermal bug, as a fixture. This must be caught."""
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        for b in range(20):
            f.write(f"[health b{b}] thermal=1.000 thermal_z=0.000 thermal_p=0.000 "
                    f"k_thermal=4.000 run_left=0.9 clock={b * 100}\n")
        p = f.name
    cols = read([p])
    os.unlink(p)
    assert not alive(cols["thermal"]), "fixture's dead sensor read as alive"
    assert not alive(cols["k_thermal"]), "fixture's dead output read as alive"
    assert alive(cols["clock"]), "a field that DOES move was read as dead"
    # and the real shape of the bug: lively during warmup, flat ever after
    warm = [2.758, 3.284, 3.499, 3.615] + [4.0] * 40
    assert len(set(warm)) > 1 and not alive(warm), "a warmup-only field read as alive"
    print("self-check OK: a dead sensor, its dead output, and a warmup-only field are all seen")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        sys.exit(main(sys.argv))
