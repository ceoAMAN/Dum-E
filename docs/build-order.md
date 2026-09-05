# Build order

The spine. Everything else hangs off a step here.

Dependency-ordered: each step unblocks the next. Doing them out of order means
building on something that returns a constant — which is how this system spent
its last life. Detail lives in the linked files.

| detail file | covers |
|---|---|
| [markov-chains.md](markov-chains.md) | expert migration + cluster sizing; the shared mechanism (step 5) |
| [cluster-territory.md](cluster-territory.md) | `tau_k`, membership test, formation vs runtime (steps 5, 7, 8) |
| [central-reliability.md](central-reliability.md) | per-cluster hallucination vector (steps 1, 2) |
| [design-rules.md](design-rules.md) | **the 22 rules the rewrite must adopt — each from a verified defect** |
| [grounded-reward.md](grounded-reward.md) | the replacement for r_i and the expert training target |
| [../errors_to_fix.md](../errors_to_fix.md) | **the single open-defect list — every error, with confidence labels** |
| [diagnosis.md](diagnosis.md) | **full-system audit — read first; it reorders the steps below** |
| [sweep-findings.md](sweep-findings.md) | 38 unverified leads from a 4-lens sweep; 3 verified by hand |
| [scope-problems.md](scope-problems.md) | the four scoped mechanisms and how each can fail |
| [open-questions.md](open-questions.md) | everything still undecided |

Claims throughout are tagged:

- **MEASURED** — a number produced by running code, with the run described.
- **DECIDED** — a deliberate choice. Change it by deciding again, not by
  patching around it.
- **OPEN** — genuinely undecided. Do not silently pick one while building.
- **PROPOSED** — a suggestion, not yet agreed.

---

## Step −1: the ownership boundary — DECIDED

Read before anything else. Several components had overlapping claims on the same
decisions; this is the split, and two steps below exist only to enforce it.

| Owner | Decides | Cadence |
|---|---|---|
| **Gating** | does this input belong to this cluster (`tau_k` test); which experts; **how many tokens each** | per input |
| **Markov chains** | which cluster an expert belongs to; what `tau_k` should be | slow, across batches |
| **Expert classes** | an expert's standing within a cluster | per batch |
| **Central reliability** | how much to trust Central's synthesis, per cluster | per batch |

The chains **feed** gating. They never allocate anything themselves.

Both of those contradict the code as it stands, which is what steps 6 and 7 are
for.

### The governing principle

Build an AK, not an HK416. Simple and reliable on every device.

Not taste — this system's own evidence. **Complex things fail silently.**
`apex_nadir_convolution.py` ran flawlessly for its entire life and returned the
constant 32, pool-wide, forever. Nothing broke. Nothing threw. Nothing measured
whether its output ever varied.

So every mechanism below is required to have a way of announcing that it has
stopped working.

---

## Step 0: `target_ids` — the root blocker

Nothing that measures quality works without a ground-truth answer `y`. Without
it: no grounded delta, no expert class standing, no Central reliability, no
hallucination profile. **Everything else waits on this.**

`data.extract_pair(example) -> {prompt, answer, verifiable}` is **written and
verified** — 8/8 configured datasets, 20/20 rows each, against live HuggingFace
rows. It is **not yet wired into `Sample` / `iter_mixture_samples`.**

What it fixed: `_extract_text` was reading **3 of 8 datasets wrongly** —
CodeAlpaca returned the prompt with the answer discarded; ai2_arc leaked the row
ID in as text and dropped the choices entirely; sciq concatenated the three
distractors alongside the correct answer with nothing marking which was which.

**Consequence:** the 20 clusters stored in `state/routing_memory.pkl` were formed
from that corrupt text. They must be re-formed (step 8), not reasoned about.

## Step 1: per-token CE

[training.py:387](../training.py) `reduction="mean"` → per-token. One line.

Mean reduction is precisely what is blind to sparse errors — a handful of badly
wrong tokens vanish into an average over hundreds of fine ones. Unblocks the
hallucination profile, free, out of the same forward pass.
See [central-reliability.md](central-reliability.md).

## Step 2: grounded delta gets a caller

`compute_grounded_r_i` ([central.py:327](../central.py)) and
`training.grounded_r_i` both have **zero callers**; `loss_deltas` is passed by
nothing.

```
delta = CE(Central(q), y) - CE(Central(q + expert_text), y)
```

A paired difference, so Central's systematic error cancels — this is what makes
Central the instrument rather than the judge.

## Step 3: `SessionTracker` persists

[memory.py:342](../memory.py). Never persisted today, so `get_current_allocation`
returns 0 and the governor's `ranked_tkl` is empty.

**This was not a defect.** Session scoping was a deliberate training-era
workaround from when experts were being constantly retrained. It is now wrong,
because expert history must survive. Change it deliberately, not as a bug fix.

## Step 4: expert classes

Per (expert, cluster) pair — per-pair because migration demands it. Four running
numbers: `n`, `sum(delta)`, `sum(delta^2)`, `sum(tokens)`.

**Score on delta-per-token, never on volume.** Class → token allocation → token
counts → class is a closed loop; scoring on totals makes it self-reinforcing, and
dormants can then never accumulate the evidence to be promoted. Rate closes it.
So does the reserved dormant trial slot.

Capped class sizes, so promotion **is** displacement.

## Step 5: chains fed real-time data

[chain.py](../chain.py) is **built and tested** (307 lines), standalone, wired to
nothing. Full design in [markov-chains.md](markov-chains.md).

Migration parameters arrive as **live runtime measurements** per (expert,
cluster) — tokens received and their frequency — not values derived from a
heuristic.

Call sites needed:

- `MigrationChains.record_move(expert_id, from, to)` wherever a move is committed
- `SizeChains.observe_load(cluster_id, share, n_clusters)` once per batch
- `SizeChains.next_tau(cluster_id)` read by gating's membership test
- `save_chains` / `load_chains` alongside `state/routing_memory.pkl`
- `reset_for_reformation()` from the offline re-formation pass (step 8)

Replaces the `self.migrations` register at [gating.py:445](../gating.py).

## Step 6: gating takes over allocation

Today [splitter.py:250](../splitter.py) sizes fragments proportional to
`convolution.compute_r_out(expert_id)`. Gating takes it, allocating on class
standing.

`schedule_by_expert`'s own docstring already assumes this — *"Gating already
knows which token spans belong to which expert"* — which it does not, yet.

## Step 7: deletions

Only after 6, because allocation is apex-nadir's last real consumer.

| delete | why |
|---|---|
| `apex_nadir_convolution.py` (720 lines) | allocation was its last consumer. It also could never rank experts: `allocation()` takes no `expert_id`, so `get_distance_to_peak` measures pool-wide sizing fit, not quality |
| `ExpertFragment.r_out`, `below_nadir`, `check_nadir_floor` | go with it |
| TKL, all three tiers | replaced by expert classes (step 4) |
| k-reduction: `select_k_cap`, `evaluate_configuration`, `recommended_x`, thermal back-off, `k_used=0` plumbing | k does not reduce in Timeline B |
| `gating.migrations` list | replaced by the chains (step 5) |
| centroid EMA at [memory.py:63](../memory.py) | conflicts with the frozen-direction rule — see [cluster-territory.md](cluster-territory.md) |

**Keep:** `FRAGMENT_MIN = 32` — an expert handed 3 tokens produces noise, so the
floor is real; it just never needed a curve-fitting engine to state a constant.
Also `memory_ceiling()` and the OOM tombstone.

The latency-intercept fix goes out with apex-nadir. It was worth making anyway:
it proved the diagnosis that a zero intercept made the quality/cost ratio
monotone decreasing, pinning `compute_r_out` to `FRAGMENT_MIN` pool-wide.

## Step 8: re-form clusters

Offline pass, on corrected text: stream → gate hidden → L2 normalise →
agglomerate at the member cut → per cluster compute the max-min vector → remove
the shared direction → **calibrate `c_k` and `sigma_k` LAST**.

Ordering is not cosmetic: calibrating before shared-direction removal collapsed
recall from 88% to 3.1%.

Re-formation invalidates every migration state index — call
`reset_for_reformation()`, do not remap.

## Step 9: fix `scripts/arch_check.py`

It asserts mechanisms are *configured to be earnable*, not that they **ran** —
line 76 asserts `not configs.EXPERT_GROUPS` as a PASS.

Currently 27/29. The two failures (`select_k_cap fired`,
`evaluate_configuration fired`) are machinery step 7 deletes, so they should be
removed rather than made to pass.
