# Rewrite map — every problem, where it is fixed

The 24 problems from the chat list and the 22 rules from
[design-rules.md](design-rules.md), each mapped to the exact place in `dume/`
that prevents it. "Prevented" means the design cannot express the defect, not
that a check catches it after the fact — checks are listed separately.

## Group A — the scoring was fake

| # | problem | fixed where | how |
|---|---|---|---|
| 1 | score could never say "worse" (floor 0.5) | `reward.score` | `d = b - a` is a signed per-token difference; Δ goes negative whenever expert text made y harder. Health canary C alarms if the negative fraction leaves 15–85%. |
| 2 | Central graded work it helped write | `reward.score` | Central appears in **both** `b` and `a_i`; Δ is the paired difference. The referent `y` came off disk; experts cannot move it. |
| 3 | honest score only ran from a script | `train.System._train_one` | There is **one** scorer. Every training batch is a dataset sample with an answer (`data.iter_mixture` only yields rows with a pair). No `target_ids` kwarg, no `if target_ids:` branch. `answer()` writes nothing. |
| 4 | spread-derived weights buried ground truth 7:1 | — | No composite. No component weights. Δ is the only reward; standing is the only ranking. There is nothing to blend. |
| 5 | experts copied Central's hidden state (1536 vs 2560) | `models.ExpertPool.update` | Expert loss is CE on the expert's **own text**, signed by Δ. No cross-model vector comparison exists anywhere in the package. |

## Group B — wrong units

| # | problem | fixed where | how |
|---|---|---|---|
| 6 | scored by domain, routed by cluster | `standing.Standing` | Counters are `(E, C)` arrays keyed by cluster id — the same id the router selects on. |
| 7 | two metrics both called TKL | — | One metric: `Standing.score` returns S_ik (nats per token). It is the only ranking quantity. TKL does not exist. |
| 8 | ranking tier read its own label back | `router.Router._pick` | Selection reads `standing.ranked(cid)`, written by `observe()` from Δ — a different event than the one being ranked. No activation-time label exists. |
| 9 | membership assigned by side effect | `standing.Standing.migrate` | `assigned` has exactly one writer, `migrate()`, called on a cadence. Running an expert writes counters, never membership. |

## Group C — numbers that could not move

| # | problem | fixed where | how |
|---|---|---|---|
| 10 | R_out always 32 (argmax of a ratio through the origin) | `router.Router._spans` | Apex-nadir deleted. Span length = composition weight × tier weight, min `SPAN_MIN`. No fitted curve, no ratio. |
| 11 | calibration invented its own measurements | — | No calibration pipeline. Reliability is measured from real `b[t]` on a held-out shard; nothing is synthesised. |
| 12 | every expert "warm" at boot from a stored flag | `standing`, `reward.Reliability` | Coldness = `n[e,c] == 0` / `N[c] < RELIABILITY_MIN_OBS`. Nothing persists a warm flag; `Reliability.R` pools cold clusters to the global estimate. |
| 13 | tau pinned by a constant 7× too small | `geometry.Geometry.form` | tau calibrated **last**, per cluster, as the 10th percentile of its own members' similarity. Then only `SizeChains.next_tau` moves it. No `min(adaptive, constant)` anywhere. |
| 14 | cluster cap keyed to a clock | `config.MAX_CLUSTERS`, `geometry.form` | Cap = `(E − √E) // √(E − √E)` = experts available per cluster. Formation is the only creator of clusters; online code cannot spawn one. |
| 15 | temperature invented from concurrency | — | No thermal term. The scheduler uses measured RAM only. If a real sensor is added later, rule 4 applies. |

## Group D — nobody in charge

| # | problem | fixed where | how |
|---|---|---|---|
| 16 | five concurrency caps | `scheduler.Scheduler` | One `k_max`, computed once from measured inputs and printed with them. `ExpertPool` has no cap of its own. |
| 17 | memory truncated routing's answer silently | `scheduler.clamp` + `router.plan` | `k = clamp(k_wanted)` happens **before** selection, and the router fills the k seats by standing — so when clamped, the most important experts are the ones kept (Aman: "when k is minimum gating prioritises important experts"). Both numbers are recorded in health (`k`, `k_wanted`). |
| 18 | allocation computed in two places | `router.Router._spans` | Gating emits `(eid, start, end)`. The train loop slices `plan.ids[start:end]` and nothing else. No splitter module. |
| 19 | three predicates, one decision | `standing.migrate` | One criterion: outperform the bottom of the target class (or the class has room). One action: `_move`. One caller. |
| 20 | no evaluation stage | `health.Health` | Every batch calls `tick()`; every mechanism reports (`delta_std`, `delta_neg_frac`, `reliability_spread`, chain accuracies, `k` vs `k_wanted`). Printed every `HEALTH_EVERY` batches. |

## Group E — geometry

| # | problem | fixed where | how |
|---|---|---|---|
| 21 | directions drifted online | `geometry.Geometry` | `B` is written once in `form()`. `set_tau` is the only online write. The unit test asserts `B` is byte-identical after `set_tau`. |
| 22 | no way to rebuild clusters | `geometry.form` + `state`/`train._restore` | Geometry carries `{gate hash, extractor version}`. At load, a hash mismatch prints INVALID and refuses to train until `form` runs again. |
| 23 | experts got sentences with holes | `router._spans` | Spans are consecutive `[start, end)` slices summing to T. Ordering by cluster centre-of-mass is how grouping happens — never by gathering indices. |

## Group F — the new mechanism

| # | problem | fixed where | how |
|---|---|---|---|
| 24 | chain.py had no event and nowhere to write | `standing.migrate` → `MigrationChains.record_move`; `train._train_one` → `SizeChains.next_tau` → `geometry.set_tau(c, …)` | The migration event exists (a `_move` between cluster ids) and the per-cluster radius exists (`geometry.tau[c]`). Both chains are observed every batch / every migration. |

## The rules, by number

| rule | where |
|---|---|
| 1 ground truth replaces proxies | `reward` — Δ is the only reward |
| 2 no variance-derived weights | no composite exists |
| 3 no curve from its own x-axis | no curve fitting exists |
| 4 loop closed only on outside signal | scheduler reads `hw.memsize` + MLX peak; no modelled sensor |
| 5 no cross-model hidden regression | expert loss is CE on own text |
| 6 one concurrency owner | `scheduler` |
| 7 one membership writer | `standing.migrate` |
| 8 one metric per question | `Standing.score` |
| 9 one lifecycle decision | `standing.migrate` |
| 10 gating emits spans | `router._spans` |
| 11 standing keyed by routing unit | `Standing` arrays are `(E, C)` |
| 12 tier keys on a different event | `ranked()` reads counters written by `observe()` |
| 13 cap is a function of the resource | `MAX_CLUSTERS`, `CENTROID_EXPERTS` |
| 14 no thresholds from different populations | tau per cluster from its own members |
| 15 direction immutable | `Geometry.B` written once |
| 16 geometry versioned and rebuilt | `form()` stamps; `_restore()` refuses a mismatch |
| 17 contiguous spans | `_spans` |
| 18 name the event and the state first | `_move` events, `geometry.tau` state — both exist before the chains observe them |
| 19 consumer before producer | reliability's consumer (`weights()`) and record (`Reliability.N/S`) exist; `ce_vector` only runs inside `score()` |
| 20 refuse rather than fall back | `scheduler` raises if `k_max < 1`; `reward.score` raises on target-length mismatch; `main train` exits 1 if nothing was measured |
| 21 coldness from n_observations | `Standing.score` returns `None` at n=0; `Reliability.R` pools below `MIN_OBS` |
| 22 one health record | `health.Health` |

## Found and fixed during verification

- **tau had no consumer** (my own rule 19). `present()` now reads it: a cluster
  is present in an input iff the input is inside its territory (`sim >= tau_c`).
  The ungrounded `PRESENT_MIN` constant is gone.
- **Load must be measured on what tau affects.** Size chains now observe the
  share of tokens actually allocated per cluster, not the composition weight —
  otherwise tightening never reduces observed load and the chain runs to TAU_MAX.
- **Surplus experts.** With fewer clusters than MAX_CLUSTERS, unseated experts
  were landing in GENERAL. They are now SURPLUS (`assigned = -2`): dormant,
  eligible only as the trial seat until they earn a cluster seat via `migrate()`.
- **The trial seat is additive.** k = present clusters + 1 (bounded by the
  sqrt(C) band), so the dormant slot fires on every batch where k_max >= 2.

## Deliberate simplifications (flag, not hide)

- **Gate backbone is frozen.** The old design LoRA'd it. Geometry is formed from
  its hidden states, so training it would move the directions (rule 15). Only the
  route head trains — by regression, so k=1 has a gradient.
- **No MAML, no lambdas.** Three losses (expert self-imitation, gate regression,
  Central pretrain CE), one term each. Nothing to meta-learn.
- **Reliability weight sign = R** (trust the instrument). One-line change in
  `reward.weights` if Aman prefers "reward the headroom".
- **Token type = centroid** (Aman p.10), not vocabulary class as the agents
  recommended. Per-token cluster assignment already exists for routing, so this
  costs nothing and lives in the same space as the decision it corrects.
- **Old expert checkpoints are not inherited** — they were trained toward the
  circular target. Experts start from base (`lora_b = 0`). Central's grounded-CE
  checkpoint IS inherited.
- **The z-discount is variance-only.** Two identical lucky batches have zero
  variance and are not discounted. A small-n prior would fix this; not added
  because it is not one of the 24 and AK says don't.
