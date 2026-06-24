# Sturnus — All Recorded Benchmarks (2026-06-24)

MacBook Air M4, 16 GB. All runs: gradients ON, 100% Timeline B during training (gate + experts learning). Numbers are measured, not projected.

---

## Table 1 — Authoritative 1M Run (post-all-fixes, every batch trains)

First complete run after all fixes including Bug 15 (the no-op-batch bug). This is the headline result.

| Metric | Value |
|--------|-------|
| Total tokens | 1,000,086 |
| Total batches | 3,629 |
| Elapsed | 10 h 52 m |
| **True throughput** | **25.6 tok/s** (full expert + 2× Central-7B passes every batch) |
| Timeline B rate | 100% (3,629 / 3,629) |
| Expert contribution r_i | 0.50 → **0.86 peak** → ~0.66 sustained |
| Routing clusters | **46 peak**, bounded 15–25 (merge/prune) |
| Final avg r_i (last 100) | 0.699 |
| MAML λ final | eff 0.282 · dom 0.288 · rel 0.146 · div 0.283 |
| avg_loss | ~0.0 (see caveat) |
| Trajectory points captured | 181 |

**Loss caveat:** `local_custom` (mixture weight 0.46) is trivial and dominates, flooring avg_loss at ~0. r_i is the meaningful learning metric here, not loss. A weight-rebalanced run (local_custom → ~0.10) would yield an informative loss curve.

---

## Table 2 — Expert Contribution r_i Learning Curve (self-supervision working)

Sampled from the 1M trajectory. r_i = cosine alignment of expert hidden state with Central's contribution delta. Rises = experts learning to contribute usefully. (Flat at ~0.49 under the pre-fix skip regime.)

| Batch | Tokens | r_i | Clusters |
|-------|--------|-----|----------|
| 20 | 5,355 | 0.499 | 9 |
| 1,020 | 273,242 | 0.552 | 23 |
| 2,020 | 554,031 | 0.708 | 21 |
| 3,020 | 824,159 | 0.632 | 15 |
| — | — | **peak 0.862** | **peak 46** |

First-10-batch mean r_i 0.502 → last-10-batch mean 0.658.

---

## Table 3 — Per-Domain K-Velocity (Core Invariant, per domain)

All four domains converge to and hold K = 1 (the floor). Decay from initial high-K occurs in the first <20 batches (faster than the 20-batch log granularity), then holds — satisfying the Core Invariant K(D,N) non-increasing **per domain**.

| Domain | Samples | Final K | K-velocity |
|--------|---------|---------|-----------|
| general | 2,000* | 1 | 0.0 (held) |
| code | 56 | 1 | 0.0 (held) |
| reasoning | 16 | 1 | 0.0 (held) |
| knowledge | 3 | 1 | 0.0 (held) |

*general capped at deque maxlen 2,000. Domain mix is skewed by local_custom (general).

---

## Table 4 — MAML Loss-Weight Emergence

The four loss-weight lambdas adapt off the uniform [0.25]×4 init and hold. (Frozen at uniform under the pre-fix regime — Bug 10.) λ_dom rises, λ_rel falls.

| Batch | Tokens | λ_eff | λ_dom | λ_rel | λ_div |
|-------|--------|-------|-------|-------|-------|
| 20 | 5,355 | 0.280 | 0.282 | 0.173 | 0.265 |
| 1,020 | 273,242 | 0.283 | 0.286 | 0.146 | 0.285 |
| 2,020 | 554,031 | 0.283 | 0.288 | 0.146 | 0.283 |
| 3,020 | 824,159 | 0.285 | 0.289 | 0.147 | 0.279 |

Controlled-run early dynamics (isolating the optimizer, network-free): λ_dom 0.268 → 0.316 ↑, λ_rel 0.222 → 0.170 ↓ over the first 65 batches.

---

## Table 5 — Throughput Across Run Lengths

| Run | Tokens | Tok/s | Note |
|-----|--------|-------|------|
| Clean 10k | 10,167 | 57 (steady) | first run after routing-memory wiring |
| Mixture 500k* | 500,075 | 528* | ⚠️ inflated (pre-Bug-15) |
| Mixture 1M* | 1,000,086 | 747* | ⚠️ inflated (pre-Bug-15) |
| **Authoritative 1M** | **1,000,086** | **25.6** | **true rate, every batch trains** |

*The 528/747 figures predate Bug 15: a confident gate emitted k=0 → ~75% of batches selected zero experts and skipped the train step. Empty batches are nearly free, inflating tok/s and meaning ~¼ effective training. The 25.6 tok/s is the honest cost of training on every token (each batch does the full expert + 2× Central-7B passes).

---

## Table 6 — Critical Bugs Found & Fixed (this audit cycle)

| # | Bug | Root cause | Fix | Evidence |
|---|-----|-----------|-----|----------|
| 9 | Gate confidence saturated at 1.000 | raw (unnormalised) backbone activations → softmax saturates; NaN checkpoint | z-score normalise → 4 domain logits; max_entropy log(4); NaN guard | untrained gate conf 0.02, rises smoothly |
| 10 | MAML lambdas frozen at 0.25 | BETA_LR=1e-5 too small under 10:1 constraint | separate LAMBDA_META_LR=0.03 + simplex floor | λ_dom 0.268→0.316 |
| 11 | 3/4 gate-loss terms zero-gradient | l_eff/l_rel/l_div computed from detached values | removed; loss = λ_dom·l_dom | grad-norm test: 0,0,0,51 |
| 12 | Routing memory never written | spawn_cluster never called in training | wired spawn + merge/prune | clusters 0 → 46 |
| 13/16 | Observability killed by adaptive block | logging sat after code that aborted each iter | observability unconditional post-increment; adaptive in try/except | [learn]/[ckpt] now fire |
| 14 | Disabled datasets opened at boot | every-key init incl. weight-0 streams | init only w>0 streams | boot opens 18 not 21 |
| **15** | **~75% of batches were no-ops** | **confident gate k=0 → select_experts returns [] → skip** | **k = max(1, k) in select_experts** | **74% skip → 0% skip** |

---

## Table 7 — Performance Audit (applied; effective next run)

Dominant cost is architectural — 2× Central-7B passes/batch (synthesis + base-reference for r_i scoring) — and is the real ~25 tok/s floor; not removable without gutting self-supervision. Removed genuine overhead around it:

| Optimization | Saving |
|--------------|--------|
| Skip lm_head + argmax + decode when not replying (training) | vocab projection 512×32k avoided/batch |
| vm_stat RAM check every 25 batches (was every batch) | one subprocess fork/batch removed |
| Removed redundant intermediate mx.eval in central.py | fewer GPU sync barriers → better pipelining |

Combined ≈ 5–15% next run. Bigger lever flagged but NOT applied (v2): batch the two Central passes into one forward.

---

## Honesty notes (carry into paper Limitations)

- Single hardware: MacBook Air M4 16 GB only. No multi-hardware validation.
- Domain mix skewed toward "general" by local_custom weight 0.46; loss floored at ~0.
- No blind Central-vs-pipeline quality benchmark yet.
- Per-domain K decay is faster than log granularity (captured as "converges to and holds K=1").

## Data files
- `analysis/validation_results.json` — machine-readable summary
- `analysis/trajectory_1m_postfix.csv` — 181 per-batch points (the authoritative run)
- `analysis/per_domain_k_1m_postfix.json` — per-domain K histories
- `analysis/plots/ri_cluster_1m_postfix.png` — r_i + cluster curves
- `analysis/plots/maml_emergence.png`, `throughput_scaling.png`, `clean_run_loss.png`
