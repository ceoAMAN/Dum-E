# Full-system diagnosis — 2026-09-04

Method: AST call-graph reachability from every entry point (module-level code of
all 15 modules + `main.py` + all 3 scripts), cross-checked with direct grep on
every headline claim. Name-resolution is by name, not by class, which
**over**-estimates reachability — so everything called dead here is dead with
margin.

**1,567 of 6,982 lines (22%) are unreachable.** Subtract `chain.py` (188, built
this session, not yet wired) and `extract_pair` (102, same) and ~1,277 lines are
orphaned code from earlier designs.

## The compound finding

Two independent P0s combine into the actual diagnosis:

> **The system trains experts, but nothing that decides which expert to use ever
> changes.**

The gate never receives a gradient. Experts never migrate. Clusters are never
created, merged or pruned. The only things that adapt are expert LoRA weights and
Central's pre-training. Every routing decision is made by a frozen random
initialisation.

That is the hollowness, mechanically.

---

## P0-1. The gate never learns

| | |
|---|---|
| **live in training.py** | `apply_expert_gradients`, `apply_central_pretrain`, `compute_central_ce`, spiderweb helpers |
| **dead** | `apply_gate_gradients` (50L), `gate_loss_fn`, `_gate_route_outputs`, `compute_l_dom`, `compute_l_eff_loss`, `compute_l_rel` |
| **also dead** | `gating.save_route_head` — so even a trained head would not persist |

`route_head` is a **frozen random init for the entire life of the system**, and
it is the entire ranking: `ROUTE_BIAS_W * route_pref` spreads 3.2e-2 against a
jitter of 1e-6, ~32,000x. Expert selection is therefore random-but-fixed.

Every reference to `apply_gate_gradients` in the repo is inside a comment.

## P0-2. No ground truth exists

`target_ids` is never produced. `data.extract_pair` is written and verified but
unwired (102L, unreachable). Consequently `training.grounded_r_i` and
`central.compute_grounded_r_i` have zero callers and cannot be given one.

Without `y`: no grounded delta, no expert class standing, no Central trust.

## P0-3. Clusters are frozen, and were built from corrupt text

| method | state |
|---|---|
| `spawn_cluster` | **DEAD** — no cluster is ever created at runtime |
| `merge_close_clusters` | **DEAD** — the 20 -> 9 merge never runs |
| `prune_stale` | **DEAD** — nothing is ever removed |
| `lookup`, `_enforce_cluster_cap`, `save`, `load`, `sync` | live |

The cluster set is whatever is in `state/routing_memory.pkl` (74KB, **dated
23 Aug**) and can never change. `lookup` still EMA-drifts centroids, so
directions move while the set is frozen — the worst combination.

Those clusters were formed before `_extract_text` was found to be misreading 3 of
8 datasets. They are 20 clusters that should be 9, built from corrupt text,
permanently.

## P0-4. The domain and migration subsystem never runs

`gating.py` — 406 of 713 lines unreachable:

- `Curriculum` (327L) — the entire class
- `DomainRegistry` (74L) — the entire class
- `next_domain`, `open_migration`, `close_migration`, `migration_delta`,
  `lifecycle`, `trial_length`, `tenure_verdict`, `next_migration_step`,
  `review_tenure`, `assign_pool`, `assign_if_ready`, `domain_thresholds`,
  `record_domain_tokens`, `seed_domain`, `target_size`

Experts never migrate between domains. Tenure is never judged. The curriculum
never advances.

**Consequence for the new design:** the Markov chain does not replace a working
register — it replaces one that never ran. `migration_delta` is dead too, so its
"unbounded list rescanned per query" flaw was never actually costing anything.
The chain is net-new capability, not a swap.

## P1-5. The memory governor and thermal control are never initialised

Dead in `diagnostics.py`: `set_memory_baseline`, `set_usable`, `observe_memory`,
`can_fit_expert`, `expert_demand`, `norm_demand`, `recommended_x`,
`temp_divisor`, `temp_multiplier`, `d_temp_multiplier`.

The governor is constructed but never fed a baseline and never observes memory,
so its ceiling is computed against nothing.

## P1-6. Calibration never runs

`main.run_calibration` is dead — its only mentions are comments, one of which
says *"its one caller was run_calibration()"* in the past tense.
`apex_nadir_convolution.fit_curves_from_calibration` (37L) is dead with it, as
are `has_calibration`, `r_out_or_bootstrap`, `experts_for`, `_eval_nadir`.

`state/calibration.npz` exists and is dated 23 Aug — same freeze as the clusters.

## P1-7. Half the data path is unreachable

`tokenize_for_gate`, `tokenize_for_expert`, `tokenize_for_central`,
`get_tokenizer`, `StreamingDataset`, `DomainLabelledStream`,
`iter_calibration_batches`, `iter_mixture_token_batches`,
`iter_group_token_batches` — 248 of 544 lines.

## P1-8. apex-nadir cannot rank experts

Known and confirmed: `allocation()` takes no `expert_id`, so
`get_distance_to_peak` measures pool-wide sizing fit, not expert quality. The
`r_out < 1e-6` branch is unreachable (`compute_r_out` ends in
`max(FRAGMENT_MIN, ...)`), so the "dead expert" guard never fires.

## P2 — the already-filed items

- `compute_central_ce` uses `reduction="mean"` — blind to sparse errors
- `SessionTracker` is never persisted
- `_enforce_cluster_cap` uses `max(10, tokens//50)`, a floor that only grows
- `chain.py` is unwired (expected)
- `scripts/arch_check.py` asserts configuration, not execution

---

## STATUS after the first implementation pass (2026-09-04)

Verified by running the system, not by reading it.

| finding | state | evidence |
|---|---|---|
| P0-1 gate never learns | **FIXED** | `route_head` weight hash changes across runs (`bc7a92e5` -> `579c73ec`) |
| P0-2 no ground truth | **FIXED** | `Sample` carries `prompt`/`answer`/`verifiable`; grounded delta overrides cosine r_i when targets exist |
| P0-3 clusters frozen | **FIXED** | `[cluster] spawned f9426183 in 'code' (r_i 0.884 > domain mean 0.863); 10 -> 6 after merge/prune` |
| P0-4 migration dead | **NOT FIXED** | `chain.py` built and tested but unwired; the old subsystem is still dead and is being deleted, not revived |
| P1-5 governor unfed | **FIXED** | `[boot] memory governor: base 268 MB, usable ceiling 9266 MB`; `can_fit_expert` now gates expert execution |
| P1-6 calibration dead | **NOT FIXED** | `run_calibration` still has no caller |
| P1-7 data path dead | **PARTLY** | `extract_pair` live; the tokenise helpers and three iterators still unreachable |
| P1-8 apex-nadir cannot rank | **NOT FIXED** | pending its deletion |
| P2 SessionTracker wiped | **FIXED** | history survives the process (`4 -> 9 activations across 6 experts`), capped at `SESSION_HISTORY_CAP` |
| P2 mean CE | **FIXED** | `compute_central_ce(..., per_token=True)` returns the un-reduced vector |
| P2 arch_check | **RESOLVED** | now 29/29. The old "29 vs 27" discrepancy was not a mystery: those two checks assert that paths *fired*, so they fail until a real run exercises them |

Unreachable code: **1,567 -> 1,249 lines** (22% -> 17%), against a codebase that
grew by ~350 lines.

**End-to-end:** `main.py --prompt ...` exits 0 and answers correctly on both
timelines.

Two defects were introduced and caught during this pass, both worth remembering:
a local `from splitter import get_available_ram_mb` inside `_timeline_b` shadowed
the module-level import and made an earlier use of the same name an
UnboundLocalError; and `Sample.verifiable` was typed as `bool` when `extract_pair`
returns the checkable answer STRING, which would have silently discarded every
exact-match target while still looking correct.

## Adversarial audit of the fixes (2026-09-04)

Seven agents, one per claimed fix, each told to REFUTE it. Result: **1 confirmed,
5 partial, 1 not actually fixed.** The claims had been over-stated. What the
audit found, and what was done about it:

| claim | verdict | the real defect | now |
|---|---|---|---|
| gate learns | CONFIRMED | — | route_head grads measured at 2.6e-2; head moves |
| ...but k=1 | — | softmax over ONE logit is constant, so `L_eff` is exactly -0.0 with zero gradient — while `l_rel` keeps `total` non-zero, so the skip diagnostic never printed. A silent no-op that logged like a healthy step. | **FIXED** — announces `k=1 — L_eff has no gradient` |
| ...and a latent crash | — | `composite` was bound only inside `if active_ids:` but read unconditionally further down. A batch with no expert output crashed the request on an unbound local. | **FIXED** — bound unconditionally |
| target_ids | PARTIAL | Extraction was real and on the live path (8/8 datasets verified). **Nothing consumed it.** | **FIXED** by the loop below |
| grounded delta | **NOT FIXED** | Every one of the 7 `.run(` call sites passes `input_text` and never `target_ids`, so `if target_ids:` was permanently False and the override was dead code behind a live-looking call site. Collateral: the Central capacity re-probe could never reach its 2-sample minimum, and `compute_r_i_batch`'s "perfection" component had no supplier. | **FIXED** — `scripts/train_grounded.py` |
| clusters | PARTIAL | Spawn/merge/prune genuinely live, but a brand-new domain could never spawn its FIRST cluster: the batch's own activations are already in the domain mean, so `r_i > domain_mean` is equal-not-greater by construction. | **FIXED** — reference drops to 0 when this batch is the domain's whole evidence |
| governor | PARTIAL | `_mem_base_mb` has **zero readers** anywhere. The boot line also claimed "after gate and central are resident" — `CentralModel.__init__` sets `model = None`, so the 4B was not resident and the number was gate-only. | **FIXED** — line now says what it actually measures |
| session tracker | PARTIAL | `token_count` was dropped by `save()`. It is the clock `VoronoiCluster.last_updated` is stamped against, so it restarted at 0 each process, `prune_stale` computed a NEGATIVE age for every carried-over cluster, and sub-floor-confidence clusters on disk were **immortal**. Also: `blob.get(...)` sat outside the try/except, so a wrong-shaped pickle crashed boot; and the history cap was never applied on load. | **FIXED** — clock persisted and monotonic, load hardened, cap applied on load |
| per-token CE | PARTIAL | Implementation correct and verified. No caller passes `per_token=True`, so the hallucination profile it was built for is still uncomputable. | **open** — needs the profile itself (step 3) |

### Proof the grounded loop works

```
[grounded] batch 1: 2 expert(s) scored against real text, r_i in [0.202, 0.245]
[gate] step 2: k=1 — L_eff has no gradient with fewer than two experts
[cluster] spawned 78dbea31 in 'reasoning' (r_i 0.593 > domain mean 0.000); 7 -> 7
```

The r_i values are the point. Cosine r_i across stored history runs **0.693-0.988**;
the same experts scored against text neither model wrote come out at **0.202-0.245**.
Those experts were being rewarded for agreeing with Central while actively making
the true answer harder to predict. That gap is the circular reward, measured.

## Fix order

This changes [build-order.md](build-order.md): **P0-1 moves to the front.**
Ground truth is useless if the thing being trained on it cannot be trained.

| # | fix | unblocks |
|---|---|---|
| 1 | wire `apply_gate_gradients` + `save_route_head` | the gate can learn at all |
| 2 | wire `extract_pair` -> `target_ids` | every quality measurement |
| 3 | per-token CE | hallucination profile |
| 4 | grounded delta gets a caller | class standing, trust |
| 5 | re-form clusters (needs `spawn_cluster` + `merge_close_clusters` wired) | routing on real geometry |
| 6 | `SessionTracker` persists | class standing survives |
| 7 | expert classes | selection, allocation, migration, dormants |
| 8 | wire `chain.py` | migration + territory |
| 9 | gating takes allocation | boundary correctness |
| 10 | delete apex-nadir, TKL, k-reduction, `Curriculum`, `DomainRegistry` | ~1,300 lines |

Steps 1-4 are what turn the system from inert to learning. Everything after is
making it learn *well*.
