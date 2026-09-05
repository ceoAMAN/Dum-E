# Sweep findings — 2026-09-04

Four independent lenses (dead code, ungrounded constants, closed loops,
contradictions) over the whole repo. A fifth (crash paths) never ran.

**These are UNVERIFIED.** The adversarial verification stage was killed by a
spend limit after 71 of 88 verifier agents failed, so nothing here has survived
a refutation attempt. Treat every entry as a lead, not a fact. Entries marked
**[VERIFIED]** were checked by hand afterwards and are facts.

Verified so far:

- **`compute_r_out` returns 32.0 for 99 of 100 experts**; `get_distance_to_peak`
  returns exactly 1.0 for all 100. Measured directly against the live state
  files. The latency-intercept fix made this session was real but insufficient —
  the apex/nadir CURVES are still never fitted, because their only writer
  (`fit_curves_from_calibration`) is dead and `run_calibration` has no caller.
  The splitter therefore sizes every fragment off a constant, and the apex-nadir
  ranking term carries zero information.
- **MAML lambdas were computed, persisted and ignored.** Every consumer
  hardcoded `configs.LAMBDA_INIT`. **FIXED** — `InferenceEngine` now takes
  `maml` and reads `maml.lambdas`. This one was introduced by my own gate
  wiring earlier today.
- **`splitter.experts_per_batch` docstring says `budget = usable / 2`** —
  "half left as working room for activations and KV cache" — and the body has
  no `/2` (`remaining / exp`). Live concurrency is **2x the documented budget**
  on the exact path that produces uncatchable Metal OOM aborts. NOT changed:
  halving concurrency is a behaviour decision, not a typo fix.

## All findings

| sev | location | finding |
|---|---|---|
| critical | `apex_nadir_convolution.py:584` | apex(t) and nadir(t) curves are never fitted on any reachable path — the only writer is dead, so the convolution reads constructor constants pool-wide |
| critical | `apex_nadir_convolution.py:467` | compute_r_out still returns the constant FRAGMENT_MIN=32 for 99/100 experts — the documented apex-nadir failure is still live |
| critical | `central.py:157` | r_i — the system's universal reward — is the expert's agreement with the shift its own output caused in Central; the grounded override is unreachable from main.py |
| critical | `training.py:203` | Experts are trained toward a target computed from their own outputs (MSE to Central's synthesis of the expert text) |
| high | `apex_nadir_convolution.py:478` | Cold-start bootstrap never happens: r_out_or_bootstrap/has_calibration have zero callers, so an uncalibrated expert reads exactly 32 tokens forever |
| high | `diagnostics.py:507` | The thermal and throughput-collapse half of the memory governor is dead: recommended_x has no caller, so four tuned config constants gate nothing |
| high | `experts.py:393` | The starvation-eviction and monopoly-overflow safety subsystem never fires — and carries recent "UNITS BUG, fixed" comments on code that has no caller |
| high | `inference.py:959` | ALLOC(T) is fitted from probe scores measured against a Central state the same experts produced, over a sample chosen by the score allocation itself generated |
| high | `inference.py:828` | MAML-adapted lambdas are computed and persisted but never consumed — every consumer hardcodes configs.LAMBDA_INIT |
| high | `memory.py:44` | RoutingMemory warm tau is structurally pinned to VORONOI_TAU_CEIL — the spread-scaled term can never bind |
| high | `splitter.py:79` | experts_per_batch: docstring reserves half the RAM budget, the body does not — the live memory cap is 2x the documented one |
| high | `splitter.py:245` | The concurrency governor still truncates the gate's expert selection, contradicting the inline claim that it no longer does |
| high | `splitter.py:356` | The gradient-mask correctness assertion never runs: validate_overlap_grads has no caller and its only consumer chain (_shadow_audit -> compute_overlap_padding) is dead |
| medium | `central.py:261` | compute_r_i_batch's grounded "perfection" component still has no supplier, contrary to the audit table in docs/diagnosis.md |
| medium | `chain.py:1` | chain.py: 307 lines, never imported by any module reachable from an entry point |
| medium | `data.py:476` | The whole calibration data pipeline is dead end-to-end — and its producer fabricates the quality scores it would supply |
| medium | `diagnostics.py:507` | Two expert-concurrency controllers; the one the config comments describe as live has zero callers |
| medium | `diagnostics.py:580` | validate_thermal_regression — a self-check that never self-checks |
| medium | `diagnostics.py:507` | The observed-peak memory governor (recommended_x) is defined but never called from any entry point |
| medium | `experts.py:314` | The anti-hallucination score measures conformity to peers, and the training step is what manufactures that conformity |
| medium | `experts.py:323` | L_div specialisation acceptance criterion is never measured: spiderweb peer-repulsion runs live but expert_weight_std has zero callers |
| medium | `gating.py:331` | gating.py: Curriculum (327L) and DomainRegistry (74L) are wholly unreachable — 401 of 726 lines |
| medium | `gating.py:704` | get_distance_to_peak returns exactly 1.0 for every expert in the pool, so the apex-nadir ranking term carries zero information |
| medium | `inference.py:887` | tkl_rank's domain-relevance tier is saturated by construction: record_activation reassigns every active expert to the batch domain before the ranking reads it |
| medium | `memory.py:320` | tkl_rank's top-priority 'domain relevance' key is a self-written label and is identically 1 for every candidate |
| medium | `memory.py:133` | The pool-derived cluster cap (MAX_CLUSTERS_BY_POOL = 10) has zero readers; the only enforced cap is a floor of 10 that grows without bound |
| medium | `memory.py:63` | Centroid EMA drift is live again and violates the DECIDED direction-freeze rule in docs/cluster-territory.md |
| medium | `splitter.py:401` | GATE_RAM_MB is documented as measured at boot "exactly as EXPERT_RAM_MB is measured"; measure_gate_ram_mb has no caller |
| medium | `splitter.py:177` | assign_spans_to_experts is unreachable; the gapped-subsequence behaviour its docstring says it fixed is still the live path |
| medium | `splitter.py:401` | Gate RAM is a hardcoded guess while expert RAM is measured: measure_gate_ram_mb is never called, and GATE_RAM_MB feeds the memory ceiling |
| medium | `splitter.py:328` | schedule_by_expert's swap-minimisation is never measured: count_expert_swaps has no caller |
| low | `central.py:346` | Superseded-but-still-present alternates: nine symbols that duplicate a live mechanism with different semantics |
| low | `configs.py:62` | MAX_SEQ_LEN is described as the ceiling on every sample read in, but the live inference path never truncates |
| low | `diagnostics.py:681` | pmset thermal proxy is a one-way ratchet: max(self._thermal_estimate, ...) means the reported temperature can never fall |
| low | `experts.py:23` | A third, hardcoded cap on concurrent experts: ExpertPool._max_loaded = 6, independent of X_MAX |
| low | `gating.py:237` | GateOutput.timeline_flag still computes the confidence rule that _is_timeline_a's docstring says it replaced, and nothing reads it |
| low | `splitter.py:177` | splitter.assign_spans_to_experts (62L) was built to replace the live splitter and never wired |
| low | `vectors.py:4` | vectors.py: 7 of 9 functions unused |

## Themes

**Closed loops (4).** Expert training targets, r_i, ALLOC(T) and the
anti-hallucination score are all reported as measuring agreement with something
the experts themselves produced. `docs/scope-problems.md` predicted this shape;
these are specific instances to check.

**Dead safety mechanisms (8).** Starvation eviction, monopoly overflow, thermal
back-off, the gradient-mask assertion, the swap-count check, the L_div
acceptance criterion, `validate_thermal_regression`, `recommended_x`. Each looks
live and never fires — the failure mode this whole audit exists to catch.

**Doc/code contradictions (6).** Several docstrings describe behaviour the body
does not implement, including the `/2` above and `measure_gate_ram_mb` being
documented as measured at boot while having no caller.

**Three independent expert-concurrency caps** (`X_MAX`, `experts_per_batch`,
`ExpertPool._max_loaded = 6`) with no single owner.
