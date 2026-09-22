# Dum-E

A 157B-parameter mixture of experts that runs on one laptop.

A Qwen2.5-0.5B gate routes over 100 Qwen2.5-1.5B LoRA experts into a Qwen3-4B
synthesiser, on an Apple M4 with 16 GB of unified memory. It fits because the
100 experts are LoRA adapters over a single frozen 4-bit base — 35 MB each, not
1.5B weights each.

The architecture is ordinary. The rule it is built under is not: **no quantity
in the system is a number anyone chose.** `k` comes from a RAM fit. The
allocation law comes from a log-log regression that refuses itself when the fit
is inadmissible. Reward is a paired cross-entropy delta against real answers,
not cosine agreement with the model's own hidden state. The span bound is a
measured backward-pass slope. Thermal regulation reads 24 real die sensors.

## Specification

| | |
|---|---|
| Gate | Qwen2.5-0.5B-Instruct-4bit, d=896 |
| Experts | 100 × Qwen2.5-1.5B-Instruct-4bit, d=1536, LoRA r=8 α=16 dropout=0.05, one frozen shared base |
| Central | Qwen3-4B-Instruct-2507-4bit, d=2560 |
| Hardware | Apple M4, 16 GB unified memory, MLX |
| k_max | 4 — `min(k_fit, k_tier, MAX_CLUSTERS)`, from a measured RAM fit |
| Clusters | 10 — `90 // floor(sqrt(90))` |
| Generals | 10 — `ceil(sqrt(100))` |
| LR / clip | 2e-05 / 1.0 |
| Expert write | 32 tokens floor, 128 target ceiling |
| State version | 7 |

Nothing in that table was picked. `k_tier = floor(sqrt(RAM_GB))`, the reserve is
`sqrt(R)` in GB, the cluster count falls out of the expert count.

## The run

3,767 batches, **639,338 tokens**, ~10.5 hours, clean tree, exit 0.

| | Previous baseline | This run |
|---|---|---|
| Tokens | 506,242 | 639,338 |
| Clone fraction | 0.25 | **0.01** |
| Routing confidence ρ | 0.326 | **0.613** |
| Standing observations | 7,001 | 9,129 |
| Reliability observations | 52,226 | 65,985 |
| Expert updates | 1,043 | 1,242 |
| Decisions | — | 2,381 admit / 945 heldout / 441 refuse |

Clone fraction is the one to read. Experts are LoRA adapters over one frozen
base, so they begin mathematically identical and can only diverge by training.
At 0.25 a quarter of batches had two experts emitting identical text. At 0.01
the pool has genuinely differentiated.

## The sensor was a constant

The thermal regulator was correct in every line, logged clean records for
thousands of batches, passed every assertion in the harness, and never once
moved `k`.

`NSProcessInfo.thermalState()` returned `fair` on all 580 samples of a
2,900-batch run, all 541 of the next, and on every direct poll under sustained
load. Its baseline converged to the one value it ever saw, its volatility sat at
0.000, its step intervals were never set because no step ever happened, and
`k_thermal` went 2.758 → 4.000 and stayed there. The device had never cast a
vote.

The silicon was never constant.

![die temperature versus the OS ordinal](docs/figures/01-sensor.png)

`models.die_temp()` reads 24 SoC die sensors through IOKit's HID event system —
match on usage page 0xff00 / usage 5, then `kIOHIDEventTypeTemperature`. No
sudo, no subprocess, no new package; pure ctypes. 16 ms for all 24 sensors,
0.03% of a batch.

The **mean** of the sensors, not the max: over a 60 s trace while training,
max(tdie) had σ=1.29 °C and 4.08 °C of range against mean(tdie)'s 0.36 and 1.13.
The max is a per-core spike detector and at one read per batch it samples noise.
Battery temperature is also sudo-free and was rejected on measurement — 31.19 →
31.20 °C over five minutes at full load.

![k_thermal over the run](docs/figures/02-k-thermal.png)

## The regulator

```
span     = peak - floor                    the range this die actually works over
z        = (T - mean) / span               how far above its own normal it sits now
pressure = max(0, z - mean|z|)             less the jitter it shows while idle
k        = k_max ** (1 / (1 + pressure * left))
                                           unless the OS reports `serious`, then k = 1
mean    += min(1, run/n) * (T - mean)      the normal re-learns; the room is not stationary
```

Four moving parts. The scale is the machine's measured working range, not its
noise floor — taken in the 0.33 °C deviation, a 3 °C rise scores 8.7σ and
collapses k to 1.05 on a die that idles at 45 and works at 67. The deadband is
the machine's own mean |z|, so idling costs nothing and it cannot lock itself
shut: z beats its own mean absolute value on ~21% of reads for anything
bell-shaped, independent of scale.

`left` is the fraction of the run still to do, and it **multiplies** the heat
rather than adding to it. Heat is a forecast — this machine will be in trouble
if it keeps working like this — and how much that matters depends entirely on
how much working is left.

![the run-fraction scaling, isolated](docs/figures/04-scaling.png)

Pressure is logged before scaling, so holding it fixed separates the scaling
from the regulator settling. Same heat late in the run keeps four times as many
experts as early, because there is almost no run left to protect.

Multiplying also settled what adding could not. A pressure in span-fractions and
a time in seconds have no common unit, and every attempt to find one either
cancelled — `seconds_left` is itself built from k, so the ratio collapses to
1/k — or pinned k at 1 for most of a long run.

## Losses

![gate and central loss](docs/figures/03-losses.png)

Read off the per-batch log lines, not off the health record. The health record
carried conditional fields forward (see below), so its loss curves plotted
smoother than the run actually was.

## What was removed

Every deletion was made because a measurement said the thing was not working,
not because it looked complicated.

| Removed | The number |
|---|---|
| The ordinal apparatus: baseline, volatility, step intervals, `excess`, the k ramp, the learned threshold | All of it existed to squeeze a rate out of a four-valued ordinal with no usable pointwise derivative. Every one was identically zero or frozen for all 2,900 batches. |
| Explicit 1st and 2nd derivatives of temperature | Built, measured, cut. On the raw reading they are noise — a *settled* machine spiked to pressure 2.06 while a real +3 °C step scored less than idle. On the mean they vanish. Audit: z carried 97–99% of the signal, the first derivative 0.8–1.9%, the second 0.4–0.7%. |
| The `k_time` term | `tau / (a + b·alloc(T))`, with tau the mean of the same latency bank `(a,b)` was fitted to. One curve divided by itself. Measured 1.08–1.14 across T=64..2048 and pinned k at 2 for all 1,566 batches. |

The derivatives were **redundant, not weak**. `z` is already the rate term:
because the mean lags, a die that has just reached 60 °C scores z=+0.504 and one
that has been at 60 °C for 400 reads scores 0.000. A fast reading against a
slowly re-learning normal is a high-pass filter whose time constant is the
machine's own history.

## What was fixed

| Problem | Measurement | Fix |
|---|---|---|
| The thermal sensor was a constant | `fair` on 100% of samples across three runs | Die temperature via IOKit; the ordinal kept only as a veto at `serious` |
| `k_thermal` read twice per batch | Every interval measured in half-batches; health logged a different k than the router used | The property caches; reporting reads the cache |
| The health record was never cleared | `graded` repeated on 100% of records, `imitate_loss` 44%, `expert_loss` 16% | Cleared after each emit; absent beats stale |
| `graded` could not report a refusal | Assigned only on the admit path. The batch lines show **441 refusals (11.7%)** the record showed none of | Written unconditionally, 0.0 by default |
| `imitate_loss` had no honest source | Recorded only into the stale record | Printed per batch with the others |
| A guard whose fixture proved nothing | A smooth `linspace` warmup leaves the deadband at **0.0000**; the mutation it was written for passed | Fixture jumps and holds, as the real die does; the guard asserts on its own fixture first |
| The run horizon reached Python by inheritance | A bare supervisor call would silently make `left` 1.0 all run | Exported; undeclared horizon gives full heat response |

Five of those seven are the same bug wearing different clothes: **something that
runs perfectly and reports nothing.** A sensor that never moves, a record that
reports history in the present tense, a field that cannot express its own
negative case, a curve with no honest source, and a test fixture that never
exercises what it tests. None is a crash. None produces a wrong number that
looks wrong. None is findable by reading the code.

The sixth happened *while fixing* the first, in the same file, on the same day.

## Verify it

```
python -m dume.main form     --samples 200     # offline geometry, once
python -m dume.main pretrain --tokens 20000    # Central alone, plain CE
python -m dume.main train    --batches 50      # the grounded joint loop
python -m dume.main run      --prompt "..."    # deployment; writes no reward

python scripts/dume_check.py                   # assertions, model-free, seconds
python scripts/did_it_fire.py                  # did each mechanism ever move?
```

`dume_check.py` is mutation-tested: each thermal assertion was confirmed to fire
when its mechanism is deliberately reverted — removing the deadband, scaling by
the noise floor, dropping the veto, freezing the mean, removing the zero floor,
restoring the double read, making `left` add instead of scale, unclamping it,
freezing the deadband after warmup, moving `graded` back to the admit path.

`did_it_fire.py` reads a run's health records and asks, per field, whether the
number ever changed. Two things make it more than a diff. **The tail** — in the
run that hid the thermal bug, `k_thermal` *did* move, 2.758 → 4.000 during
warmup as its baseline converged, then never again; so every field is judged on
the last half of the run too. **The chain** — a constant output is only a bug
when its input moved; when the input is dead too, it says look upstream, which
is where the fault was.

On the final run: 753 records, **0 problems**, all six chains ok.

## Layout

```
dume/            the system
  chain.py       thermal regulator, migration and size chains
  models.py      gate, expert pool, central, die_temp()
  scheduler.py   RAM fit, k_max, residency
  alloc.py       the allocation law and the cost model
  router.py      planning, splitting, k_effective
  train.py       form / pretrain / train / answer
  reward.py      paired-CE deltas, reliability
scripts/
  dume_check.py      assertions
  did_it_fire.py     dead-mechanism detector
  build_figures.py   the figures above
  build_report.py    the long-form report
  fresh_run.sh       archive, re-form, launch
analysis/        archived run summaries and the run report
docs/            design notes; paper-original-2026-06.md is the superseded paper
```

## Known and deferred

**Expert-pass amortisation.** A pass costs `a = 0.542 s` fixed plus
`b = 0.00507 s/token`, and experts write 8–29 tokens, so ~84% of a pass is fixed
cost. Irrelevant to training (17% of a batch, dominated by the backward passes)
and material to deployment, where the answer path has no backward pass and the
sequential expert loop sits entirely in front of the first token. Before any fix:
split that 0.542 s into park / load_weights / prefill. Those are three different
problems with three different fixes.

**`imitate_loss` for this run.** 44% carried forward. 425 genuine measurements
survive and the curve is recoverable in shape — and that shape is flat, 2.356 →
2.166 across the whole run. What is lost is the firing rate, which is a count and
does not need 639k tokens to measure.

**Per-expert budget.** ~151 selections per expert across the run. This is a
mechanism test, not a capability run.

**`MIGRATE_EVERY = 20`** suspected thrashing, unexamined. **A `state.VERSION`
bump** silently discards clock, batch and consumed.
