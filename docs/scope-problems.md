# Scope as described, and the problems inside it

> Companion to [build-order.md](build-order.md). Four mechanisms as scoped, each
> with the specific way it can fail.

## 1. Gating: matrix dot product assigns the domain

Per-cluster vectors `v_k` stacked into a matrix `V`, input `x` dotted against it,
result says which domain(s) the input belongs to.

### Problem 1a — it is a decomposition, not an assignment

"Assigns domain" reads as picking one. The design is a **membership test**: an
input can belong to several clusters, or to none. That was deliberate — an
argmax router is invariant to a pedestal, so a shared direction across centroids
cannot be detected by it, and the whole shared-direction removal step becomes
pointless.

So `V x` produces a **composition** — how much of this input lives in each
domain — not a label. Every downstream consumer must be written for a vector.
Anything that takes `argmax(V x)` quietly reinstates the router and discards the
work.

### Problem 1b — a moving `tau_k` destroys cross-cluster comparability

`tau_k` is now per-cluster and adapts (see
[cluster-territory.md](cluster-territory.md)). Membership per cluster is still
well defined. **A composition is not**, because each cluster's score is being
compared against a different, drifting threshold — so the weights are on
incomparable scales and do not mean the same thing from one batch to the next.

**Resolution — DECIDED by construction:** read the composition off the **frozen
`v_k` directions only**. `tau_k` gates membership and routing; it must never
enter the composition weights. Directions are fixed at formation, so a
composition measured at batch 100 and batch 5000 mean the same thing.

## 2. Central post-synthesis composition, trust, and the losses

Central's synthesis output is itself projected to a vector; its composition tells
us what went into the answer; weighted by trust, that drives expert loss, gate
loss, and how much heavy lifting each expert is given.

### Problem 2a — provenance circularity. This is the serious one.

If the post-synthesis composition reflects **which experts ran**, the signal is
worthless and actively harmful:

```
gate routes to cluster 5
  -> only cluster-5 experts contribute
    -> Central's output is composed of cluster-5 material
      -> composition reads "cluster 5"
        -> gate is trained toward "cluster 5 was right"
```

The gate is trained on a restatement of its own decision. It self-confirms,
disagreement sits near zero forever, and every metric looks healthy. Same shape
as Central-as-judge, one level up.

**The composition must be read from CONTENT, not provenance** — project Central's
output onto the frozen cluster directions, independently of which experts were
loaded. Then a gate that routed to cluster 5 while the answer's content lies
along cluster 3 produces real disagreement.

Even done correctly this measures **consistency, not correctness** — gate and
Central can be wrong together. It is a cheap secondary signal. The grounded delta
against `y` stays the primary one.

### Problem 2b — trust must gate action, not learning

If low trust down-weights the loss, the system learns **least** where it is
**worst**. That is backwards. Low trust is a reason not to act on an output and a
reason to learn harder from it.

Trust weights inference-time use. It should not scale the training loss.

### Problem 2c — "was this composition trained well?" is not countable as posed

Composition is a continuous vector. You cannot count how often you have seen one.
Answering this needs an explicit discretisation — bucket, or nearest of a fixed
set of reference compositions — and the buckets are arbitrary until chosen.
**OPEN.**

## 3. The `sqrt(c)` bracketing constants

General case: take `c^(1/2)`; on a decimal, take the **successor** (round up).
Then over all centroids minus general: `c^(1/2)`, take the **predecessor** (round
down).

Partly present already: `nl(n)` is the lower bracket, `NU(E) = ceil(sqrt(E))` the
upper, at [training.py](../training.py) and [gating.py](../gating.py). The
existing justification for the asymmetry is the right shape — round **up** for a
grace period (evict only on thicker evidence), **down** for a selection rule.

### `c` = EXPERTS — PINNED

Derived from `EXPERT_POOL_SIZE`, so the numbers follow the pool rather than being
typed in:

```
general      = ceil(sqrt(N))          # successor on a decimal
rest         = N - general
per_centroid = floor(sqrt(rest))      # predecessor
```

| pool N | general | rest | per-centroid |
|---|---|---|---|
| **100 (live)** | **10** | **90** | **9** |
| 64 | 8 | 56 | 7 |
| 25 | 5 | 20 | 4 |
| 9 | 3 | 6 | 2 |
| 4 | 2 | 2 | 1 — degenerate |

### Problem 3b — RESOLVED

The small-cluster degeneracy worry assumed `c` counted cluster *members*, where
`floor(sqrt(2)) = 1`. With `c` = experts it cannot arise: `floor(sqrt(90)) = 9`.
Degeneracy only appears at N <= 4, which is not a real pool. A guard at N < 9 is
enough.

### Problem 3c — DOES NOT APPLY here

The warning was that a formula does not make a number adaptive. It aims at
quantities that are *supposed* to vary — `compute_r_out` was meant to differ per
expert and returned 32 for all 100.

These are **structural caps**, deliberately fixed. A cap that holds still is
working. The warning stands for anything meant to adapt; it does not apply to
these two.

### Problem 3d — NEW. The rule never references the cluster count

`per_centroid` is computed from the expert pool alone, so the per-centroid caps
sum past the pool once there are enough clusters:

| clusters | slots | vs 90 non-general experts |
|---|---|---|
| 9 | 81 | fits |
| **10** | **90** | **exactly full — the boundary** |
| 11 | 99 | oversubscribed by 9 |
| 20 | 180 | oversubscribed by 90 |

Re-formation is expected to produce ~9 clusters, so this fits with **one
cluster of headroom and no margin.**

Whether that is a problem depends on an unsettled question — see below.

### Problem 3e — DECIDED. Membership is exclusive; standing is per-pair.

Two parts of the design currently disagree:

- `MigrationChains.home[expert_id]` in [chain.py](../chain.py) is a **single**
  cluster per expert — exclusive.
- Expert classes are per **(expert, cluster) pair** — implying an expert has
  standing in several.

**DECIDED.** An expert *belongs* to exactly one cluster (`home`,
which migration changes) but *has standing* in every cluster it has been tried
in. Home is exclusive; scoring is per-pair. They are then consistent, and the
9-per-centroid cap applies to home membership.

That reading makes 3d a real constraint: **the cluster count is capped at 10 by
the expert pool.** Cluster formation and territory resizing cannot spawn past it
without either shared membership or a larger pool.
