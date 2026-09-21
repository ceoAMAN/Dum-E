# run_20260921_b2900 — the flat-128 baseline

2900 batches, **506,242 tokens**, finished 2026-09-21 09:31:42 after ~10 h.
Code at `4eef0dc` (branch `symbiosis-curriculum`), untouched for the whole run.

This is the LAST run of the old prompt regime and exists to be compared against.
Every expert prompt here carried the input's first `TARGET_MAX_TOKENS = 128`
tokens above its own span. Since T has median 51 and p90 639, that meant the
whole input for four rows in five.

## What it is a baseline for

`span-head` replaces that construction: the expert sees its fragment, a map of
the whole written by the frozen gate, and a position line. Compare against the
**late window** of this run, not its mean — see below.

## Numbers

| | |
|---|---|
| ADMIT / HELDOUT / REFUSE | 1826 / 715 / 359 |
| distinct experts trained | 100 / 100 |
| expert updates | 1043 (update_frac 0.62) |
| delta_neg_frac (final) | 0.810 |
| reliability observations | 52,226 |
| thermal | level 1 throughout, vol 0.000, k_thermal 3.997, k pinned at 4 |

`clone_frac` by fifth: **0.316  0.306  0.298  0.280  0.234**

It FELL, monotonically, 26% across the run. An earlier diagnosis at b1555 called
it flat; that was true of the window measured and not of the series. The prompt
redundancy it was attributed to is separately measured and real — the k prompts
were 84-93% identical and the pool cost k*T — but "the expert update is not
doing anything" was never true.

## Contents

- `summary.json` — the above, machine-readable
- `batches.csv` — 2900 rows: decision, k, k_wanted, home, rho, per-expert deltas, losses
- `health.csv` — 580 records, every health field
- `standing_b2900.npz` — standing arrays at the final batch
- `commits.txt` — the six commits this run was built on
- `raw/state_b2900.pkl` — full state, restorable
- `raw/dume-fresh-500k.log`, `raw/supervisor.log`
- `ckpt_experts/` — 100 expert adapters + central (3.6 GB, git-ignored)

## Restore

```bash
cp analysis/run_20260921_b2900/raw/state_b2900.pkl state/dume/state.pkl
rm -rf state/dume/ckpt && cp -R analysis/run_20260921_b2900/ckpt_experts state/dume/ckpt
```
