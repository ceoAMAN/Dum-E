# Cluster territory — `tau_k`

> Steps 5, 7 and 8 of [build-order.md](build-order.md).

How much of the input space a centroid claims, and how that changes as traffic
frequency changes. Implemented as `SizeChains` in [chain.py](../chain.py).

## The geometry — DECIDED

In the membership-test routing design a cluster's territory is a spherical cap:

```
territory(k) = { x : v_k . x >= tau_k }        |x| = 1
```

Its area is monotone in `tau_k` alone. So **resizing a cluster is one number.**

That number is exactly the membership cut that was left open ("0.90 costs ~13%
recall"). A hand-set global cut is an ungrounded constant closing an adaptive
loop — this system's most-repeated failure. Driving it from observed traffic
**removes** the constant rather than tuning it.

## Direction freezes, radius breathes — DECIDED

`v_k` is fixed at formation. `tau_k` moves at runtime.

Every formation-time guarantee — the max-min vector, shared-direction removal,
calibrating `c_k` and `sigma_k` LAST — is a statement about *direction*. Letting
`v_k` drift online voids all of them with no symptom. The radius is the one
degree of freedom that is safe to move.

Changing directions is what the offline re-formation pass is for. Re-formation
also resets the migration chains (see [markov-chains.md](markov-chains.md)).

### Collision to resolve before wiring — OPEN

[memory.py:63](../memory.py) currently EMA-drifts the centroid on **every**
assignment:

```python
best_cluster.centroid = configs.EMA_DECAY * best_cluster.centroid + ...
```

So direction already moves at runtime today. If direction drifts *and* radius
adapts, the formation geometry quietly stops holding. This EMA has to go, or be
reconciled with the freeze rule. It cannot just be left running.

## Feedback sign — DECIDED, and it is the whole design

**High traffic → TIGHTEN.**

| rule | behaviour |
|---|---|
| high traffic → widen | runaway. more territory → more traffic → more territory → one cluster eats the sphere |
| high traffic → tighten | self-correcting. an overloaded cluster is too coarse, so it narrows and overflow lands on neighbours; a starved cluster widens to catch more |

This is load balancing, not rich-get-richer.

## Load regimes

Traffic share is bucketed against what an equal split would give
(`share * n_clusters`):

| regime | relative share | `tau` step |
|---|---|---|
| STARVED | < 0.25x | −2 |
| LIGHT | 0.25–0.75x | −1 |
| HEALTHY | 0.75–1.5x | 0 |
| HEAVY | 1.5–3x | +1 |
| OVERFULL | > 3x | +2 |

Steps are in units of `TAU_STEP = 0.005`, hard-clamped to
`[SIM_NEIGHBOUR = 0.70, TAU_MAX = 0.97]` — inside the empirically validated band
structure, so a mispredicting chain can make a cluster somewhat too tight or too
loose but never make it swallow the sphere or vanish.

The signal is already being collected, incidentally: `sample_count` at
[memory.py:70](../memory.py) counts exactly this, then gets spent on a confidence
that saturates at 50 and never decays.

**MEASURED**, 5,000 rounds each: OVERFULL 0.900 → 0.970 (tighten), STARVED
0.900 → 0.700 (widen), HEALTHY holds at 0.900. All clamps hold.

## Why a chain rather than a control law — DECIDED

A reactive rule tightens *after* the cluster is already overloaded — it lags by
a batch, every time. One-step-ahead lets the adjustment land before the
overload, and `accuracy()` says whether the prediction is worth acting on at
all.

## Conservation is NOT enforced — OPEN, watch this

Clusters are coupled: tightening one pushes its rejects onto its neighbours.
`SizeChains` does not enforce conservation of total territory. The clamps bound
the damage instead — the AK choice over a constrained optimiser.

**The symptom to watch for is inputs that match NO cluster.** That is what
over-tightening looks like from the outside, and it needs a counter at the
gating call site. If it climbs, either raise `SIM_NEIGHBOUR`, or add explicit
conservation, or spawn a cluster.
