# Run archive — b3425, 596,215 tokens

Everything measured up to the point the run was stopped on 2026-09-20. Kept for
the paper. **The live state was reset after this snapshot**, so these numbers are
the only record of the pre-reset regime.

## What produced it

Two runs against the same state: b153–b1852 (1700 batches) and b1859–b3441
(1583 batches, log in `raw/dume-final-500k.log`, stopped by SIGKILL at b3441;
state is the b3425 checkpoint). `commits.txt` lists the code, newest first.

Important: this run measured the machinery under the **pre-fix** equations. The
defects found in the 2026-09-20 audit were all live during it. Treat every
number below as "what the old equations produced", which is exactly what makes
it a useful baseline.

## Files

| file | contents |
|---|---|
| `summary.json` | final state: clusters, tau, domain, reliability, elite, seats, alloc fit |
| `batches.csv` | 1583 rows: batch, source, timeline, k, k_wanted, M, rho, verdict, losses, per-expert deltas |
| `health.csv` | 316 health records, every recorded field |
| `standing_b3425.npz` | raw standing matrices: n, sd, sn, sc, sh, ssec, stok, assigned, elite |
| `raw/state_b3425.pkl` | the full pickled state, restorable |
| `raw/*.log` | the run logs, including two July runs on the old architecture |

## Headline numbers

- 980 ADMIT / 421 HELDOUT / 182 REFUSE; 720 Timeline A / 863 Timeline B
- 210.79 nats saved against 637.85 hallucinated -> measured w_halluc **3.03**
- 4 clusters; reliability R = [0.405, 0.626, 0.393, 0.470]; 57,082 observations
- standing 4566 observations, 543 migrations, 36 seated + 10 general

## Defects live during this run (all verified, see memory/dume-mechanism-audit.md)

1. `rate @ d` scored an unmeasured cluster 0.0, which beat 202 of 222 real
   (negative) cells. pearson(overall, clusters measured) = -0.283, p=0.0044.
2. The general class was locked at election and took **0 of 3550 observations**;
   52 of 100 experts never ran.
3. `k_time` self-normalised, pinning k at 2 in every batch while the law asked
   for 11-114.
4. SizeChains could not reach a tightening regime (max rel 1.194 vs cut 1.50),
   so tau never moved.
5. Imitation teachers had a negative delta in 359 of 470 steps (76.4%).
6. 86% of deltas negative is mostly headroom, not expert damage:
   spearman(central_loss, delta) = +0.243; CE<0.3 -> 90.7% negative,
   CE>=2.0 -> 44.4%.
7. The MAD floor was effectively constant (0.0180 +/- 0.0034); 47% of admitted
   batches took no gradient.
8. Health alarms latched: 170 of 399 ALARM U lines printed after the condition
   had cleared. **Do not read the log's alarm lines as present tense.**
9. Throughput is not a property of the expert: one shared decode cost of
   41.9 tok/s at R2 0.978; the 2.3x spread correlates -0.026 with output length.

## Restoring

    cp raw/state_b3425.pkl ../../state/dume/state.pkl

## Expert adapters

`ckpt_experts/` holds the 87 trained expert adapters from this run (2.8 GB,
git-ignored). They were MOVED out of `state/dume/ckpt/` so the fresh run starts
from a uniform cold pool — `ExpertPool.load()` finds no checkpoint and falls
back to `_fresh()`, which re-draws lora_a per expert.

Why they were not kept: they were trained under the broken imitation (359 of
470 teachers had a negative delta) and the constant MAD floor, and they are
UNEVEN — 48 experts received roughly ten gradient steps and 52 received none.
A claim that the architecture learns should not start from that.

Central (63 MB) and the gate (352 KB) were kept: the gate's weight hash stamps
the geometry, and re-forming would need a fresh model pass.

Restore:

    mv analysis/run_20260920_b3425/ckpt_experts/expert_* state/dume/ckpt/
