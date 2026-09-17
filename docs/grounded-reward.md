# The grounded reward

Replaces r_i (`central.py:157`) and the expert training target (`training.py:203`),
both verified circular. Two independent agents designed replacements from the same
constraints; they converged on the core and **disagree on one sign**, flagged below.

## The three quantities

Per grounded sample `(q, y)` routed to cluster `k`, experts `i = 1..m` with output text
`e_i` and emitted token count `n_i`. All CE is per-token over the target region only
(`compute_central_ce(..., per_token=True)` — already implemented and verified).

Truncate `y` to `M = limit//4` **once, before any pass**, so both passes score the same
`y_1..y_M`.

```
b[t]   = CE_t( Central, q | y )              baseline — ONE forward, shared by the batch
a_i[t] = CE_t( Central, q | e_i | y )        one forward per expert
d_i[t] = b[t] - a_i[t]                       nats saved on real target token t
```

`b` and `a_i` are element-wise comparable because both index the same `y`; only the
context before them differs.

### 1. Central reliability — a vector, per cluster

```
c(t) = TYPE[y_t]     static int array over Central's vocab, built once offline
                     {number, operator/code-punct, identifier, whitespace/structure,
                      common-word (top-2k), rare/other}
N_k[c] += 1 ;  S_k[c] += b[t]
R_k[c]  = exp( -S_k[c] / N_k[c] )    in (0,1]
```

Updated **only on a held-out shard** (`hash(sample_id) % 4 == 0`); the other 3/4 feed
standing. Disjoint by construction, so R is never fitted on the tokens it grades.

### 2. Batch delta — the number that replaces r_i

```
Δ_i = Σ_t R_k[c(t)] · d_i[t]  /  Σ_t R_k[c(t)]        nats per target token
```

**This is where the vector does work a scalar provably could not.** With a scalar R it
factors out of numerator and denominator identically for every `i` and cancels exactly.
With a vector, the weights multiply an expert-**specific** profile `d_i[t]`.

### 3. Expert standing — per (expert, cluster), delta-per-token

```
n += 1 ;  ΣΔ += Δ_i ;  ΣΔ² += Δ_i² ;  ΣN += max(n_i, 32)
Var    = ΣΔ²/n - (ΣΔ/n)²
S_ik   = ( ΣΔ - z·√(n·Var) ) / ΣN ,   z = 1
```

Units are a **rate**: doubling output for the same total help halves S, so volume cannot
buy standing and the class → allocation → token-count → class loop is broken **at the
divisor**. The z-discount means two lucky batches cannot outrank fifty consistent ones.

Four counters — the same four `build-order.md` step 4 already specifies. No new state.

### 4. Admission gate

```
ρ = (1/T) Σ_t R_k[c(t)] ;   admit the batch only if ρ ≥ R_MIN
```

If Central cannot predict this kind of text at all, it is not a usable instrument here
and Δ is noise, not a small number. **Refuse to measure rather than measure badly.**

## The expert weight update

Replaces the MSE. Central is frozen while the batch is scored and is never differentiated through; its own CE step on `y` runs last in the batch, after the experts are graded.

Expert `i` emits `G ≥ 2` candidates `e^1..e^G` of its fragment (temperature ~0.8). Score
each by `s^g = Δ^g / n_g`. Then:

```
A^g = (s^g - mean_g s) / (std_g s + eps)
L_i = - Σ_g A^g · (1/n_g) · Σ_j log p_θi( e^g_j | fragment, question, e^g_<j )
```

Ordinary next-token CE on the expert's **own sampled text**, signed and scaled by how
much that text lowered Central's CE on `y`. No ratios, no clipping, no KL, no reference
model. If all candidates score alike, `std → 0`, `A → 0`, and **no update happens** —
the mechanism silences itself instead of inventing a direction.

Fallback if variance is a problem (RAFT form, same signal): take `g* = argmax_g s^g` and
apply plain CE only when `s^{g*} - mean_g s > τ`.

**Delete, do not port:** the MSE-to-synthesis term, the compatibility/direction cosines
as a training signal, `spiderweb_target` weight-space attraction, `peer_weight_vector`
repulsion. All are agreement terms with no external referent, and measurement already
shows they did nothing (`l_div_sim` nonzero in 6 of 152 rows).

## Why this is not circular

Everything that varies across the experts being ranked is the presence or absence of
`e_i` in a prompt scored against a **fixed `y` that came off disk**. The experts cannot
move the referent — only be more or less useful for predicting it.

Formally: hold the expert's output fixed and swap `y` for a different real answer, and Δ
changes sign. Under `training.py:203` the same swap changes nothing at all, because `y`
does not appear in the expression.

Central appears identically in `b` and `a_i`, so Δ is a **paired difference** and
Central's systematic error cancels to first order. That is "instrument, not judge": the
old design read Central's absolute opinion; this reads only how much Central's error
**changed**.

Consequences the old reward could not have:

- A confidently wrong expert **raises** Central's CE on `y`, so `Δ < 0`, `A < 0`, and the
  gradient pushes it away from that text. Under the MSE, the same text became part of its
  own target and was reinforced.
- An expert that merely agrees with Central earns **nothing**: `a_i ≈ b`, `Δ ≈ 0`.
- The reward has a hard ceiling set outside the models: `Δ ≤ Σ_t R·b_t`.

**Residual coupling, not claimed away:** Central is shared, so if it is systematically bad
at a whole domain, every Δ there compresses toward zero. That is a *sensitivity* limit,
not a circularity — the reward is weak there, not self-confirming. The admission gate is
what stops that being mistaken for "no expert helped".

---

## THE ONE DISAGREEMENT — needs Aman's call

The two designs weight token classes in **opposite directions**.

| | weight | rationale |
|---|---|---|
| **A — trust the instrument** | `w_c = R_k[c]` (high where Central's CE is **low**) | Where Central predicts badly, Δ is noise. Weight by inverse noise: discount credit earned where the instrument can't read. |
| **B — reward the headroom** | `w_c = ρ_c / mean(ρ_c)`, `ρ_c` = mean CE (high where Central is **bad**) | Credit is worth more on classes Central is independently bad at — that is where an expert adds value. |

Both are defensible and they produce opposite rankings on the same data.

**I lean A.** B re-introduces a Goodhart target: experts chase the classes Central is
worst at, which are exactly the classes where Δ is least trustworthy. A keeps Central
strictly an instrument. But B is closer to "what do we actually want experts to fix", and
that is a product decision, not a measurement one.

## Cost

Measured baseline: 994,231 tokens / 9,350 batches at 135.9 tok/s = **0.78 s/batch**;
k = 1.0028 (765 of 765 `[learn]` lines show k=1).

Measured on this machine (36GB M-series, 4-bit): Central full forward with logits
0.44s @128 tok / 0.87s @256 / 1.74s @512. Expert generation 0.336s for 16 tokens.

Per grounded step, 1 scored expert, G=4 candidates, ~256 tok context:
5 Central forwards (1 shared baseline + 4) ≈ 4.4s, expert generation 4 × 0.336 ≈ 1.4s,
LoRA backward ≈ 0.15s. **≈ 5.9s.**

Amortised at 1 expert every 4th batch: **+1.5 s/batch**, ~2.9× the old wall clock.
`G=2` halves the Central cost (+0.65 s/batch, ~1.8×). That is the only real knob.

**None of it is on the user's latency path** — grounded batches run in dead time from
the dataset stream. Today `dead_time_orchestrator` replays pending Timeline-A *text*,
which by definition has no `y`; it should replay `iter_mixture_samples()` instead.

**Memory:** no new resident weights. The one allocation to watch is the CE logits tensor:
`V = 151,936`, so full-sequence fp32 at T=512 is **~0.31 GB**. Score candidates
**sequentially**, never as a `(G, L)` batch — that makes it 1.2GB+ and is exactly where a
silent Metal OOM comes from. Slice to the target rows before the fp32 cast.

**State:** 4 floats per (expert, cluster) + 2 per (cluster, token-type) ≈ under 10k floats.

---

## How it fails silently, and the check for each

Every one prints on a fixed cadence. **A mechanism that cannot announce its own death is
the one that emitted 32 forever.**

| # | silent failure | check |
|---|---|---|
| A | **y never arrives** — the current bug in new clothes. Standing counters never advance while the run looks healthy. | Every 100 batches print `Σn` over all pairs and assert strictly increasing. A run reporting 0 grounded measurements exits non-zero. |
| B | **Both forwards see the same context** — `_build_input_ids` silently drops expert text when the question fills the limit, so `a_i == b` and every Δ is 0. Reads as "experts don't help", which is plausible and therefore dangerous. | Assert `len(ids_with) > len(ids_without)`; assert the context grew by exactly `n_i`; alarm if >5% of batches have `max|d_i| < 1e-6`. |
| C | **Δ degenerates to a band** — the exact pathology being replaced (r_i was 0.7572 ± 0.0522, std collapsing 0.1160 → 0.0130 by batch 2000). | Rolling std of Δ over 200 batches, alarm below 0.01 nats. Plus the **sign test**: fraction of `Δ < 0` must sit in 30–60%. A scorer that never returns a negative is not measuring anything — cosine r_i produced 1 value below 0.5 in 916. |
| D | **Tracks something other than correctness** — survives A–C because every number still moves. | On the 4 datasets carrying `Sample.verifiable`, split batches by whether Central's greedy continuation exact-matches the string, and require mean Δ on the matched half > unmatched, over ~200 samples, nightly. **Model-free** — string vs dataset field, no component participates, so no agreement loop can fake it. Gate on this one. |
| E | **The reliability vector flattens** — every `R_k[c]` converges, at which point it cancels like the scalar it replaced. | Log `max(R_k) − min(R_k)` per cluster, alarm below 0.05. Alarm if any bucket has `N_k[c] < 100`; pool to the global R until it fills. |
| F | **Held-out leakage** — the hash split breaks and R is fitted on what it grades. | Assert shard disjointness at each update; report R on both shards. If indistinguishable, the split isn't working. |
| G | **Pre-merge gate** (`scripts/delta_harness.py`, already written, not yet used) | Fixes three expert outputs per case — true / irrelevant / confidently wrong — and asserts `Δ(true) > Δ(irrelevant)`, `Δ(true) > Δ(wrong)`, `spread(Δ) >> spread(cosine)`. Needs no training run. **Would have caught the entire cosine regime on day one.** Run in CI on every change to the scoring path. |

One more, worth an explicit assert during the rewrite: `contribution_norm` is logged as
`inf` on the final rows of `logs/benchmarks.csv`. That metric goes away with the cosine
path, but it is a live example of a non-finite value flowing through a metric channel
unnoticed for an entire 994k-token run. **Finite-check every scalar at the point it is
recorded, not at the point it is read.**
