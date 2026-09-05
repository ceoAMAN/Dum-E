# Markov chains — expert migration and cluster sizing

> Step 5 of [build-order.md](build-order.md).

Implemented in [chain.py](../chain.py) (307 lines). Standalone and tested; not
yet wired to anything.

Two things in this system change slowly and must be **remembered** rather than
recomputed: which cluster an expert belongs in, and how much space a cluster
claims. One mechanism, two mountings.

## Why a chain and not a register — DECIDED

A register already exists: [gating.py:445](../gating.py) `migration_delta(from, to)`
keeps an append-only list of migration records and rescans it on every query.

It fails on three counts:

1. **No conditioning on the present.** It averages over every expert that ever
   made that move. Every expert asking the same question gets the same answer.
2. **No forward generation.** It reports what happened. It cannot say where an
   expert is heading.
3. **Unbounded.** The list grows forever and is rescanned per query.

A chain conditions (`P(next | current state, this expert)`) and generates
forward. The accumulated counts are not a summary kept *alongside* the past —
they *are* the past, in the only form that predicts.

Storage: one `S x S` float matrix. Update O(1). No history buffer. Same
sufficient-statistic move the latency fit and the class scores use.

## Horizon rule — DECIDED, load-bearing

**One step. No stationary distributions, ever.**

The chain adapts on purpose — new observations keep moving the transition
probabilities. That makes it **non-stationary**, and every attractive Markov
result (stationary distribution, equilibrium populations, long-run shares)
assumes transitions hold still. They do not here.

`P^n` for large `n` would always compute, always look reasonable, and never
signal that it was meaningless. That is the apex-nadir failure mode exactly.
`chain.py` deliberately exposes no such function.

Short horizons also buy the self-check below: a one-step prediction can be
compared against what actually happened next.

## The three properties, and what each costs

### Bounded evidence — `CHAIN_MEMORY = 500.0`

With pure accumulation a new observation moves the estimate by `1/n`: 2% at
n=50, 0.01% at n=10,000. The chain **freezes** — still running, still confident,
no longer responsive, no symptom.

A row accumulates freely to `CHAIN_MEMORY`, then rescales before each increment.
Asymptotically an EMA with `lambda = 1 - 1/M`, but better cold (early
observations count fully) and stated as evidence rather than a decay rate:
*"this chain remembers 500 transitions."* You can say that in English; you
cannot say `lambda = 0.998` in English.

Note this satisfies "we don't wipe history" — old evidence is diluted, never
deleted, and 10,000 transitions and the matrix they produced predict identically.

**MEASURED.** After 10,000 observations of A→B, truth flips to A→C. Observations
until the chain notices:

| memory | observations to switch |
|---|---|
| M = 500 | **347** |
| M = 5,000 | 3,465 |
| uncapped | 10,001 |

### A prior instead of a floor — `CHAIN_PRIOR = 1.0`

Every cell starts at `alpha` rather than zero, so a row with three observations
returns near-uniform — *"no opinion"* — instead of a confident estimate off three
samples.

This replaces an `if evidence < threshold: abstain` guard at every call site. A
guard gets forgotten at one of them. A prior cannot.

**MEASURED.** 0 obs → `[.25 .25 .25 .25]`, evidence 0.0. 51 obs → `[.018 .018
.018 .945]`.

### Self-check — free accuracy meter

Every `observe()` scores the standing prediction against what actually happened
*before* folding in the new evidence. `accuracy()` is then a live report on
whether the chain is worth listening to. If a cluster re-forms underneath it, or
a state definition drifts, the hit rate falls and says so.

**MEASURED.** Learnable pattern (true p=0.8): recovers 0.808, accuracy 0.794.
Pure noise on 4 states: accuracy 0.243 — it announces its own uselessness.

This is the property the rest of the system has lacked.

## Mounting 1: expert migration

State = cluster index. Parameters are fed as **real-time data at runtime** —
live measurements per (expert, cluster), not values derived from a heuristic.
Per the original framing: updated on tokens received and their frequency.

**Pool chain carries the signal; per-expert chains refine it.** This is not a
nicety. An individual expert migrates a handful of times in its entire life, so
a per-expert `S x S` matrix is prior-dominated for most of its existence.
`evidence(i)` is what tells you which of the two is actually speaking, and
`predict_home` blends them weighted by exactly that.

`CHAIN_SEED_STRENGTH = 8.0` — a fresh (expert, cluster) pair inherits the pool's
typical behaviour worth ~8 pseudo-observations, then gets overruled by its own.
Directly serves the dormant story: a cheap trial starting from "this is how
experts generally move here" converges in far fewer trials than one starting
from a coin flip.

**MEASURED.** Fresh expert inherits the pool's answer; its own evidence
overrules after **14 moves at C=6 and 13 at C=20** — i.e. independent of cluster
count, which was the point of the second fix below.

**Cluster re-formation resets migration chains outright.** State *indices* stop
meaning the same thing — "cluster 7" afterwards is a different region of space —
so `reset_for_reformation()` discards and prints loudly rather than remapping.
`resize()` exists but is only safe when indices keep their meaning.

## Mounting 2: cluster sizing

See [cluster-territory.md](cluster-territory.md).

## Two defects found by testing, both instructive

1. **Seed strength scaled with state count.** `strength * rows * n_states` meant
   the pool grew harder to outvote every time a cluster was added — ~14 real
   moves to overrule at C=6, ~40 at C=20. A parameter whose meaning silently
   drifts with an unrelated quantity. Now absolute.

2. **A cold pool manufactured belief.** Seeding at a flat `strength` from a pool
   with *zero* evidence gave a fresh expert 8 pseudo-observations of support for
   a uniform guess — and `predict_home` returned a confident-looking cluster
   recommendation, confidence 0.167, built on nothing. Now each row transfers at
   most what that row has really seen; cold pool → confidence 0.000.

The second is *"an ungrounded constant closing an adaptive loop"* — this
system's most-repeated bug — appearing inside brand-new code written expressly
to avoid it. Only a test that asked the cold case found it. **Test the cold
case.**

## API

```python
c = MarkovChain(n_states)     # prior + memory from configs
c.observe(i, j)               # scores standing prediction, then records
c.predict(i) -> np.ndarray    # P(next | i). one step.
c.next_state(i) -> int
c.evidence(i) -> float        # real observations, prior mass removed. 0 = prior only
c.accuracy() -> float | None  # live self-check
c.seed_from(pool, strength)
c.resize(n)                   # only when indices keep their meaning

mg = MigrationChains(n_clusters)
mg.record_move(expert_id, from_cluster, to_cluster)
mg.predict_home(expert_id) -> (cluster, confidence)   # confidence 0.0 = no opinion
mg.reset_for_reformation(n_clusters)

save_chains(path, migration, sizing) / load_chains(path)
```

`predict_home` returns `(-1, 0.0)` for an expert with no home on record. Callers
must treat confidence `0.0` as *no recommendation*, not as a weak one.
