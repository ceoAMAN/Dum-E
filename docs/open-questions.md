# Open questions

> Companion to [build-order.md](build-order.md).

Undecided. Do not silently pick one while wiring — each changes the build.

## Design

1. **What is a "token type"** for Central's hallucination vector? Vocabulary
   class / position in the answer / direction in cluster space. I lean the
   third (reuses the routing geometry). See
   [central-reliability.md](central-reliability.md).

2. **Reliability as confidence or as subtraction?** Subtraction is defensible
   now that it is a vector rather than a scalar. Confidence is cleaner. Not
   settled.

3. **Centroid EMA at [memory.py:63](../memory.py)** — conflicts with freezing
   `v_k` at formation. Delete, or reconcile? Cannot be left running.

4. **Territory conservation.** Not enforced; clamps bound the damage instead.
   Needs a counter on inputs matching NO cluster before this is safe to trust.

5. **The membership cut's starting value.** `tau_k` now moves, so this matters
   less, but the cold value is still `SIM_MEMBER = 0.90`, which cost ~13% recall
   in the synthetic test. The sweep: 0.80 → 98.5%, 0.85 → 95.1%, 0.90 → 87.2%,
   0.95 → 72.3%; false-admit 0.000% throughout (synthetic data separates too
   cleanly for that number to mean much).

6. **Does a neighbour-band hit move a centroid?** Unresolved from the routing
   design.

7. **`_enforce_cluster_cap` contradicts the pool cap.** [memory.py:133](../memory.py)
   is `cap = max(10, token_count // CLUSTER_CAP_RATE)` — a FLOOR of 10 that only
   grows, so a 1M-token run permits ~20,000 clusters. Exclusive membership caps
   the count at `MAX_CLUSTERS_BY_POOL = 10`. Needs to become a `min`, or be
   replaced outright by the step 8 re-formation pass. Not patched now: that whole
   path is being rewritten and a fix here would be churn.

## Risks to watch, not blockers

8. **Migration circularity.** An expert receives cluster-7 traffic *because* it
   is assigned to cluster 7. If transition evidence is dominated by that, the
   chain learns "everyone stays put" and migration silently freezes — with
   `accuracy()` looking *excellent*, because "no move" is easy to predict.

   Migration parameters are fed as real-time data, so this is not a design
   blocker. But **high accuracy plus near-zero off-diagonal mass is the
   signature** of it, and it should be checked once real data flows.

9. **Dwell time.** A dormant that has been dormant 50 batches differs from one
   just demoted. If transitions depend on dwell, the process is semi-Markov and
   the estimate is biased. Cheap fix if needed: split states by dwell bucket
   (`dormant-fresh` / `dormant-stale`).

10. **Two dormant systems.** `spiderweb` ([training.py:255](../training.py))
   already does dormant *improvement* via weight-space attraction. The chain and
   classes do dormant *selection*. Complementary — keep it that way.

11. ~~**`arch_check` 29/29 vs 27/29**~~ **RESOLVED 2026-09-04.** Not a mystery:
    `select_k_cap fired` and `evaluate_configuration fired` assert that those
    paths actually EXECUTED, so they fail on a clean checkout and pass once a
    real run has exercised them. Now 29/29.

12. Mid-loop unload at [inference.py:391](../inference.py); `EXPERT_RAM_MB`
    weights-vs-peak; `1/n` EMA step scaling elsewhere in the codebase (the same
    freeze the chain's evidence cap fixes).

## Answered, recorded so they are not reopened

- **k does not reduce.** Timeline A or B is chosen up front; inside B, k holds.
- **Expert history is not wiped.** Session scoping was a training-era artifact.
- **The chain does not allocate.** Gating does.
- **No stationary distributions.** The chain is non-stationary by construction.
- **Sets are not clusters.** Datasets are provenance; clusters form
  unsupervised, and labels are a purity check only.
